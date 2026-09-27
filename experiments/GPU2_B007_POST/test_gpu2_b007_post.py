from __future__ import annotations
import importlib.util, sys, unittest
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("post_utils", HERE / "post_utils.py"); utils = importlib.util.module_from_spec(spec); assert spec and spec.loader; sys.modules[spec.name] = utils; spec.loader.exec_module(utils)

class PostTests(unittest.TestCase):
    def frame(self, reverse=False):
        rows = [("Q1","S2-A","S2","IN",1,.9),("Q2","S3-B","S3","US",0,.1)]
        if reverse: rows.reverse()
        return pd.DataFrame(rows, columns=utils.REQUIRED)
    def test_alignment_is_keyed_not_positional(self):
        a,b,report=utils.align_frames(self.frame(),self.frame(True),"a","b"); self.assertEqual("PASS",report["status"]); self.assertTrue(a[utils.KEYS].equals(b[utils.KEYS]))
    def test_alignment_rejects_label_difference(self):
        b=self.frame(); b.loc[0,"label"]=0
        with self.assertRaisesRegex(ValueError,"labels differ"): utils.align_frames(self.frame(),b,"a","b")
    def test_alignment_rejects_duplicates(self):
        a=pd.concat([self.frame(),self.frame().iloc[[0]]])
        with self.assertRaisesRegex(ValueError,"duplicate"): utils.align_frames(a,self.frame(),"a","b")
    def test_probability_grid_math(self):
        self.assertAlmostEqual(.82,utils.probability_mix([.8],[1.0],.1)[0])

if __name__ == "__main__": unittest.main()
