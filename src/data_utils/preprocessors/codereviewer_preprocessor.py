from .default_preprocessor import DefaultPreprocessor


class CodeReviewerPreprocessor(DefaultPreprocessor):
    def _preprocess_diff(self, diff: str, line_sep: str, **kwargs) -> str:
        """Helper method: add tags from CodeReviewer to single file diff."""
        processed_lines = []
        for line in diff.split(line_sep):
            line = line.strip()
            if not line:
                continue
            if line.startswith("+"):
                processed_lines.append("<add>" + line[1:])
            elif line.startswith("-"):
                processed_lines.append("<del>" + line[1:])
            else:
                processed_lines.append("<keep>" + line)
        return line_sep.join(processed_lines) + line_sep
