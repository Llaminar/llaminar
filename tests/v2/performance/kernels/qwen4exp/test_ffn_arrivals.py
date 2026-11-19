"""Check verifier layer attribution across staged and fused native FFN schedules."""

from contextlib import closing
import sqlite3
import unittest

from ffn_arrivals import participant_rounds


class VerifierFFNArrivals(unittest.TestCase):
    """Replay the profiler fields used to delimit prefill, FFN layers and verification rounds."""

    def trace(self, producers, width=128):
        """Include a warmup that reuses the measured graph IDs and two measured rounds."""
        db = sqlite3.connect(":memory:")
        db.executescript("""
            CREATE TABLE kernel_dispatch(agent_id, graph_exec_id, kernel_id, start, end);
            CREATE TABLE graph_launch(agent_id, graph_exec_id, kernel_dispatch_count);
            CREATE TABLE info_kernel_symbol(id INTEGER PRIMARY KEY, display_name TEXT UNIQUE);
        """)
        prefill = ["preparePrefillChunkViewKernel", "prefillHead"]
        target = []
        for producer in producers:
            target.extend(["peerPushFP32Kernel", "groupedFFNFormatProgramGate", producer])
        target.append("summarize_greedy_speculative_verify")
        db.executemany("INSERT INTO graph_launch VALUES(1, ?, ?)", [(1, len(prefill)), (2, len(target))])
        start = 0
        for graph, symbols in ((1, prefill), (2, target), (1, prefill), (2, target), (2, target)):
            for name in symbols:
                db.execute("INSERT OR IGNORE INTO info_kernel_symbol(display_name) VALUES(?)", (name,))
                db.execute("""INSERT INTO kernel_dispatch
                              SELECT 1, ?, id, ?, ? FROM info_kernel_symbol WHERE display_name=?""",
                           (graph, start, start + 4, name))
                start += 10
        tables = {name: name for name in ("kernel_dispatch", "graph_launch", "info_kernel_symbol")}
        with closing(db):
            return participant_rounds(db, tables, 1, 2, [width] * len(producers))

    def test_completion_of_each_production_schedule(self):
        result = self.trace(["groupedFFNTokenDown", "rocm_moe_shared_expert_gate_add_kernel",
                             "groupedFFNFormatProgramDown<15u, true, 7, 128>"])
        self.assertEqual(result, [[(164, 20), (194, 20), (224, 20)],
                                  [(264, 20), (294, 20), (324, 20)]])

    def test_gate_alone_cannot_certify_a_layer(self):
        with self.assertRaisesRegex(ValueError, "Missing or extra FFN producer"):
            self.trace(["groupedFFNFormatProgramGate"])

    def test_native_reader_width_checks_the_inventory(self):
        self.trace(["groupedFFNFormatProgramDown<15u, false, 19, 0>"], width=256)
        with self.assertRaisesRegex(ValueError, "shard width differs"):
            self.trace(["groupedFFNFormatProgramDown<15u, true, 7, 256>"], width=128)


if __name__ == "__main__":
    unittest.main()
