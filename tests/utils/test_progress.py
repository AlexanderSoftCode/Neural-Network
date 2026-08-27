import io
import re
import unittest

from aether._utils.progress import TrainingProgress, make_progress, _visible_len

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_MAX_TRAILING = 16


def _overwrite(segments):
    """Applies carriage-return overwrite semantics to one rendered line."""
    buf = []
    for seg in segments:
        buf[0:len(seg)] = list(seg)
    return "".join(buf)


def _drive(progress, epochs=3, steps=40, log_every=8):
    """Runs a full multi-epoch lifecycle, capturing everything written."""
    sink = io.StringIO()
    progress._write = sink.write
    progress._flush = lambda: None

    losses = [1.5, 0.87, 1.4321, 0.9, 1.05]
    accs = [0.5, 0.8125, 0.6, 1.0, 0.0625]
    lrs = [9.62e-4, 1e-3, 5e-5, 1.234e-6, 8.1e-4]

    for epoch in range(1, epochs + 1):
        progress.start_epoch(epoch)
        for step in range(1, steps + 1):
            progress.tick(step, force=True)
            if step % log_every == 0 or step == steps:
                i = step % len(losses)
                progress.update_metrics(
                    step, losses[i], accs[i], lrs[i], reg_loss=0.0007 + i * 1e-4
                )
        progress.commit_epoch(epoch, 1.1, 0.5, lrs[epoch % len(lrs)])
        progress.commit_validation(0.9 - epoch * 0.05, 0.6 + epoch * 0.02)
    progress.close()
    return sink.getvalue()


class TestJupyterFrameCoverage(unittest.TestCase):
    """The padded frame must cover its predecessor without a visible band."""

    def setUp(self):
        self.stream = _drive(
            TrainingProgress(40, 3, has_reg=True, render_for="jupyter")
        )

    def _rendered_lines(self):
        for raw_line in self.stream.split("\n"):
            if "\r" in raw_line:
                yield raw_line

    def test_no_residue_under_either_overwrite_model(self):
        for raw_line in self._rendered_lines():
            for model, strip in (("raw-index", False), ("column", True)):
                segments = [
                    _ANSI.sub("", s) if strip else s for s in raw_line.split("\r")
                ]
                stranded = _overwrite(segments)[len(segments[-1]):]
                self.assertEqual(
                    stranded.strip(),
                    "",
                    f"{model}: previous frame left stranded: {stranded[:80]!r}",
                )

    def test_committed_lines_carry_no_whitespace_band(self):
        for raw_line in self._rendered_lines():
            visible = _ANSI.sub("", _overwrite(raw_line.split("\r")))
            trailing = len(visible) - len(visible.rstrip())
            self.assertLessEqual(
                trailing,
                _MAX_TRAILING,
                f"{trailing} trailing spaces on {visible.rstrip()[-40:]!r}",
            )

    def test_frozen_frame_keeps_its_telemetry(self):
        # Jupyter draws bar and telemetry as one line, so the epoch summary
        # lands beneath the frame rather than consuming it.
        frozen = [
            _ANSI.sub("", _overwrite(line.split("\r")))
            for line in self._rendered_lines()
        ]
        self.assertTrue(frozen, "no live frames were rendered")
        for line in frozen:
            self.assertIn("100%", line)
            self.assertIn("loss", line)
            self.assertIn("lr", line)

    def test_epoch_summary_follows_each_frozen_frame(self):
        summaries = [ln for ln in self.stream.split("\n") if "Total]" in ln]
        self.assertEqual(len(summaries), 3)


class TestOtherRenderModes(unittest.TestCase):
    """tty and plain paths must not inherit the jupyter padding scheme."""

    def test_tty_erases_with_ansi_rather_than_padding(self):
        stream = _drive(TrainingProgress(40, 2, has_reg=True, render_for="tty"))
        self.assertIn("\033[2K", stream)
        for raw_line in stream.split("\n"):
            visible = _ANSI.sub("", raw_line.split("\r")[-1])
            trailing = len(visible) - len(visible.rstrip())
            self.assertLessEqual(trailing, _MAX_TRAILING)

    def test_plain_emits_no_escapes_and_no_carriage_returns(self):
        stream = _drive(TrainingProgress(40, 2, has_reg=True, render_for="plain"))
        self.assertNotIn("\r", stream)
        self.assertNotIn("\033", stream)
        self.assertIn("Epoch 1/2", stream)


class TestVisibleLen(unittest.TestCase):
    def test_discounts_sgr_runs(self):
        self.assertEqual(_visible_len("\033[92m▲1.25%\033[0m"), 6)
        self.assertEqual(_visible_len("plain"), 5)


class TestMakeProgress(unittest.TestCase):
    def test_verbose_zero_is_silent_null_object(self):
        progress = make_progress(0, 10, 1)
        self.assertEqual(type(progress).__name__, "_NullProgress")

    def test_verbose_two_forces_plain(self):
        self.assertEqual(make_progress(2, 10, 1).render_mode, "plain")

    def test_null_progress_parity(self):
        # The silent path calls the display unconditionally, so any method
        # added to TrainingProgress needs a counterpart on _NullProgress.
        null = make_progress(0, 10, 1)
        live = TrainingProgress(10, 1, render_for="plain")
        public = {
            name for name in vars(TrainingProgress)
            if not name.startswith("_") and callable(getattr(live, name))
        }
        missing = sorted(name for name in public if not hasattr(null, name))
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
