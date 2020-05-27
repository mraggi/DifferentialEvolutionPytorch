from fastprogress.fastprogress import progress_bar as fpb

class VariableSizeProgressBar(fpb):
    def on_update(self, val, text, interrupted=False):
        self.total = len(self.gen)
        super().on_update(val,text,interrupted)

progress_bar = VariableSizeProgressBar
