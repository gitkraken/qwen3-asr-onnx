import os
import shutil
import tempfile
from pathlib import Path


def publish_artifacts(replacements, removals=()):
    backup = Path(tempfile.mkdtemp(prefix=".release-backup-", dir=replacements[0][1].parent))
    saved = []
    installed = []
    clean = False
    try:
        targets = [target for _, target in replacements] + list(removals)
        for index, target in enumerate(targets):
            if target.exists() or target.is_symlink():
                old = backup / str(index)
                saved.append((old, target))
                os.replace(target, old)
        for source, target in replacements:
            installed.append(target)
            os.replace(source, target)
        clean = True
    except BaseException:
        try:
            for target in reversed(installed):
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
            for old, target in reversed(saved):
                if old.exists() or old.is_symlink():
                    os.replace(old, target)
            clean = True
        except BaseException as recovery_error:
            raise RuntimeError(
                f"Publication recovery was interrupted; inspect the destination files and backup directory: {backup}"
            ) from recovery_error
        raise
    finally:
        if clean:
            shutil.rmtree(backup)
