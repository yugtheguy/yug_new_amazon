import sys
import importlib.util

packages = ['numpy', 'pandas', 'scipy', 'sklearn', 'lightgbm', 'rapidfuzz', 'joblib', 'unidecode']
missing = []
versions = {}

for p in packages:
    if importlib.util.find_spec(p) is None:
        missing.append(p)
    else:
        try:
            module = __import__(p)
            versions[p] = getattr(module, '__version__', 'unknown')
        except Exception:
            missing.append(p)

if missing:
    print('Missing:', missing)
    sys.exit(1)
else:
    print('All imports successful')
    for p, v in versions.items():
        print(f"{p}: {v}")
