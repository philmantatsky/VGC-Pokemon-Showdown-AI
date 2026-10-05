"""Opponent action predictor (side experiment; design: OPPONENT_PREDICTOR.md).

Foundation modules:

* ``events``       - pure-string log reading: turn segments and the per-slot
  ``SlotAction`` labels (``split_log``, ``segment_turns``, ``read_turn_actions``),
  dex-derived move predicates and intent classes, account ids (``user_id``).
* ``public_state`` - ``PublicBattle`` (event-fed public state), ``drive_log``
  (offline: snapshots and labels per turn) and ``LiveShadow`` (the same state
  from a live battle's ``_replay_data``).

Two things the caller owes these modules, because no stream can supply them:
the team-sheet state of a player-view stream (``sheets`` / ``sheets_known``,
``LiveShadow.feed_sheet`` / ``mark_sheets_known``), and a check of
``events.dex_signature()`` between the dataset and the runtime.

Nothing here is imported by the deployed bot. Importing this package loads no
poke-env or torch code.
"""
