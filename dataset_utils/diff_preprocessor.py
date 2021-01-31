from dataset_utils.edit_distance_utils import align_lists
        prev_lines, updated_lines, removed, added = [], [], [], []
        removed_idxs, added_idxs = [], []
        rename_len = None
        another_file_name_len = None

        for i, tokens_in_line in enumerate(tokens_per_line):
                # name of changed file
                # example: mmm a / telecomm / java / android / telecomm / Connection . java
                # example (if new file was created): mmm / dev / null
                if tokens_in_line[1] == 'a':
                    prev_lines.append(tokens_in_line[3:])
                    another_file_name_len = len(tokens_in_line) - 3
                else:
                    prev_lines.append(tokens_in_line[2:])

                # name of changed file
                # example: ppp b / telecomm / java / android / telecomm / Connection . java
                # example (if file was deleted):  ppp / dev / null
                if tokens_in_line[1] == 'b':
                    updated_lines.append(tokens_in_line[3:])
                    if prev_lines[-1] == ['dev', '/', 'null']:
                        prev_lines[-1].extend(["<empty>" for _ in range(len(tokens_in_line) - 3)])
                else:
                    updated_lines.append(tokens_in_line[2:])
                    cur_len = len(tokens_in_line) - 2
                    updated_lines[-1].extend(["<empty>" for _ in range(another_file_name_len - cur_len)])

                # line in git diff when new file is created
                # example: new file mode 100644

                # line in git diff when file is deleted
                # example: deleted file mode 100644

                # line in git diff when file was renamed (old name)
                # example: rename from src / forge / resources / worldedit . properties
                if rename_len:
                    cur_len = len(tokens_in_line) - 2
                    if cur_len < rename_len:
                        prev_lines[-1].extend(["<empty>" for _ in range(rename_len - cur_len)])
                    else:
                        updated_lines[-1].extend(["<empty>" for _ in range(cur_len - rename_len)])
                    rename_len = None
                else:
                    rename_len = len(tokens_in_line) - 2

                # line in git diff when file was renamed (new name)
                # example: rename to src / forge / resources / defaults / worldedit . properties
                if rename_len:
                    cur_len = len(tokens_in_line) - 2
                    if cur_len < rename_len:
                        updated_lines[-1].extend(["<empty>" for _ in range(rename_len - cur_len)])
                    else:
                        prev_lines[-1].extend(["<empty>" for _ in range(cur_len - rename_len)])
                    rename_len = None
                else:
                    rename_len = len(tokens_in_line) - 2

                # line in git diff when file mode was changed
                # example: old mode 100644
                # 644=rw-r--r--

                # line in git diff when file mode was changed
                # example: new mode 100755
                # 755=rwxr-xr-x

                # lines that were removed
                # example: - version = ' 2 . 0 . 2 '
                removed.append(tokens_in_line)
                removed_idxs.append(i)

                # lines that were added
                # example: + version = ' 2 . 0 . 3 '
                added.append(tokens_in_line)
                added_idxs.append(i)

                # some special info that we are not interested in
                # example 1: index 0000000 . . 3f26e45
                # example 2: similarity index 100 %

                # all other cases
                # case 1: line that was not changed (do not drop them)
                # case 2: Binary files a / dependencies / windows / sumatra / SumatraPDF . exe and / dev / null differ
        # align removed and added
        removed, added = align_lists(removed, added)

        for (i, prev_line), upd_line in zip(enumerate(removed), added):
            if len(removed_idxs) > len(added_idxs):
                prev_lines.insert(removed_idxs[i], prev_line)
                updated_lines.insert(removed_idxs[i], upd_line)
            else:
                prev_lines.insert(added_idxs[i], prev_line)
                updated_lines.insert(added_idxs[i], upd_line)


