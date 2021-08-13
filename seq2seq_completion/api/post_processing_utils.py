import re


class PostProcessor:
    @staticmethod
    def remove_spaces(x: str) -> str:
        x = re.sub(" *([" + re.escape("!\"#$%&'()*+,-.:;<=>?@[]^_`{|}~/" + r"\\" + "\n") + "]) *", r"\1", x)
        return x

    @staticmethod
    def add_spaces_right(x: str) -> str:
        pattern = "([,!\?:;\)])([^\s/,!\?:\.;\)'\"`])"
        x = re.sub(pattern, r"\1 \2", x)
        x = re.sub(" +", " ", x)
        return x

    @staticmethod
    def add_spaces_left(x: str) -> str:
        pattern = "([^\[\(\s])([#])([\d])"
        x = re.sub(pattern, r"\1 \2\3", x)
        x = re.sub(" +", " ", x)
        return x

    @staticmethod
    def fix_quotes(x: str) -> str:
        pattern1 = "(['\"`\[][^'\", \`]*?['\"`\]])([,:;!\?.]|$)"
        pattern2 = "(['\"`\[][^'\", \`]*?['\"`\]])([^,:;!\?.]|$)"
        x = re.sub(pattern1, r" \1\2", x)
        x = re.sub(pattern2, r" \1 \2", x)
        x = re.sub(" +", " ", x)
        return x

    @staticmethod
    def process(x: str) -> str:
        x = PostProcessor.remove_spaces(x)
        x = PostProcessor.add_spaces_right(x)
        x = PostProcessor.add_spaces_left(x)
        x = PostProcessor.fix_quotes(x)
        return x.strip(" ")
