from pathlib import Path


def _same_file(first, second):
    return first.exists() and second.exists() and first.samefile(second)


def _same_entry(first, second):
    if not _same_file(first.parent, second.parent):
        return False
    if first.name == second.name:
        return True
    if not _same_file(first, second):
        return False
    names = {entry.name for entry in first.parent.iterdir()}
    # Replacing one hardlink entry preserves other entries with the same inode.
    return first.name not in names or second.name not in names


def _inside(path, directory):
    return _same_entry(path, directory) or any(_same_file(parent, directory) for parent in path.parents)


def paths_overlap(first, second):
    first, second = Path(first).absolute(), Path(second).absolute()
    resolved_first, resolved_second = first.resolve(), second.resolve()
    return (
        resolved_first.is_relative_to(resolved_second)
        or resolved_second.is_relative_to(resolved_first)
        or _inside(first, second)
        or _inside(second, first)
    )


def _access_paths(path):
    path = Path(path).absolute()
    for _ in range(41):
        yield path
        for entry in [*reversed(path.parents), path]:
            if entry.is_junction():
                raise ValueError(f"Junction in source access path: {entry}")
            if entry.is_symlink():
                yield entry
                link = entry.readlink()
                target = link if link.is_absolute() else entry.parent / link
                path = target / path.relative_to(entry)
                break
        else:
            return
    raise ValueError(f"Too many symlinks in source access path: {path}")


def _destination_entry(path):
    path = Path(path).absolute()
    return path.parent.resolve() / path.name


def validate_publication_sources(reads, *, files=(), directories=()):
    targets = [(_destination_entry(path), False) for path in files]
    targets.extend((_destination_entry(path), True) for path in directories)
    for source in reads:
        for path in _access_paths(source):
            for target, directory in targets:
                conflict = _inside(path, target) if directory else _same_entry(path, target)
                if conflict:
                    raise ValueError(f"Publication would replace a source access path: {source} through {target}")
