import collections
import normalization as norm

COMMON_ADDR_WORDS = {
    'street', 'road', 'avenue', 'boulevard', 'drive', 'court', 'lane', 'place',
    'circle', 'way', 'trail', 'parkway', 'highway', 'suite', 'floor', 'apartment',
    'building', 'room', 'number', 'near', 'opposite', 'behind', 'block', 'sector',
    'phase', 'plot', 'flat', 'door', 'fl', 'no', 'unit', 'north', 'south', 'east', 'west',
    'india', 'us', 'usa', 'france', 'state', 'district', 'city', 'nagar', 'colony',
    'bazaar', 'marg', 'gali', 'null', 'rd', 'st', 'ave', 'blvd', 'dr', 'ct', 'ln',
    'hwy', 'apt', 'ste', 'first', 'second', 'third', 'ground'
}


def get_blocking_keys(name, addr, country):
    """
    Generates multi-attribute blocking keys from business name and address.
    """
    c_n, core_n, sort_n = norm.normalize_name(name)
    c_a, nums, sort_a = norm.normalize_address(addr)

    keys = set()
    n_tokens = core_n.split()
    a_tokens = c_a.split()

    # 1. Compact name (no spaces)
    compact_name = ''.join(n_tokens)
    if len(compact_name) >= 4:
        keys.add(('compact_n', compact_name))
        keys.add(('compact_sort_n', ''.join(sorted(n_tokens))))

    # 2. Core name exact
    if len(core_n) >= 3:
        keys.add(('core_n', core_n))

    # 3. First 2 tokens of core name & sorted tokens
    if len(n_tokens) >= 2:
        keys.add(('n2', f'{n_tokens[0]}_{n_tokens[1]}'))
        keys.add(('sort_n', ' '.join(sorted(n_tokens))))
    elif len(n_tokens) == 1 and len(n_tokens[0]) >= 3:
        keys.add(('n1', n_tokens[0]))

    # 4. Individual name tokens
    for t in n_tokens:
        if len(t) >= 4:
            keys.add(('n_tok', t))

    # Address tokens: extract significant words
    sig_addr_words = [t for t in a_tokens if t not in COMMON_ADDR_WORDS and len(t) >= 3 and not t.isdigit()]

    # 5. Number + City (last 2 tokens of address)
    if nums and len(a_tokens) >= 1:
        for num in nums:
            if len(num) >= 1:
                keys.add(('num_city1', f'{num}_{a_tokens[-1]}'))
                if len(a_tokens) >= 2:
                    keys.add(('num_city2', f'{num}_{a_tokens[-2]}'))

    # 6. Street number + rare address word
    if nums and sig_addr_words:
        for num in nums:
            for w in sig_addr_words[:3]:
                keys.add(('num_street', f'{num}_{w}'))

    # 7. Name token + Street number
    if n_tokens and nums:
        for num in nums:
            keys.add(('name_num', f'{n_tokens[0]}_{num}'))
            if len(n_tokens) >= 2:
                keys.add(('name_num2', f'{n_tokens[1]}_{num}'))

    # 8. Name token + City
    if n_tokens and len(a_tokens) >= 1:
        keys.add(('name_city', f'{n_tokens[0]}_{a_tokens[-1]}'))
        if len(a_tokens) >= 2:
            keys.add(('name_city2', f'{n_tokens[0]}_{a_tokens[-2]}'))

    # 9. Pairs of rare address words
    if len(sig_addr_words) >= 2:
        for i in range(min(len(sig_addr_words), 4)):
            for j in range(i + 1, min(len(sig_addr_words), 4)):
                w1, w2 = sorted([sig_addr_words[i], sig_addr_words[j]])
                keys.add(('addr_pair', f'{w1}_{w2}'))

    return keys


def build_inverted_index_for_country(target_records, name_prune=300, addr_prune=150):
    """
    target_records: dict of tid -> (name, addr, country)
    Returns: inverted index dict: key -> list of tids
    """
    index = collections.defaultdict(list)
    for tid, (rname, raddr, rcountry) in target_records.items():
        keys = get_blocking_keys(rname, raddr, rcountry)
        for k in keys:
            index[k].append(tid)

    # Prune ultra-frequent keys
    for k in list(index.keys()):
        limit = name_prune if (k[0].startswith('n') or k[0].startswith('core') or k[0].startswith('compact')) else addr_prune
        if len(index[k]) > limit:
            del index[k]

    return index


def retrieve_candidates_for_s1(rname, raddr, rcountry, inverted_index, top_k=50):
    """
    Given an S1 record and the country inverted index, returns [(tid, shared_key_count), ...]
    """
    s_keys = get_blocking_keys(rname, raddr, rcountry)
    counts = collections.Counter()
    for k in s_keys:
        if k in inverted_index:
            counts.update(inverted_index[k])
    if counts:
        return counts.most_common(top_k)
    return []
