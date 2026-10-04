"""League persistence and eviction: members in use are never evicted, and a moved run still loads."""

import shutil

from vgczero import data as D
from vgczero.league import League
from vgczero.model import ModelConfig, VGCNet


def _tiny():
    return VGCNet(ModelConfig.preset("tiny"))


def test_eviction_keeps_members_in_use(tmp_path):
    D.load()
    lg = League(tmp_path / "league", max_size=3)
    m = _tiny()
    ids = [lg.snapshot(m, u, tags=["anchor"] if u == 0 else None) for u in range(3)]
    for i in ids[1:]:
        lg.members[i].ema = 0.9  # well beaten: eviction candidates
    in_use = ids[1]
    lg.members[in_use].ema = 0.99  # the most beaten, so it would go first
    new = lg.snapshot(m, 3, keep={in_use})
    assert in_use in lg.members and new in lg.members
    assert len(lg.members) == 3
    lg.model(in_use, "cpu")  # still loadable


def test_moved_league_loads(tmp_path):
    D.load()
    lg = League(tmp_path / "a" / "league", max_size=8)
    mid = lg.snapshot(_tiny(), 0)
    shutil.move(str(tmp_path / "a"), str(tmp_path / "b"))
    lg2 = League(tmp_path / "b" / "league", max_size=8)
    assert mid in lg2.members
    lg2.model(mid, "cpu")
    (tmp_path / "b" / "league" / lg2.members[mid].path).unlink()
    assert mid not in League(tmp_path / "b" / "league", max_size=8).members
