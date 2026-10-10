#!/usr/bin/env bash
# Exit 0 when this laptop's lid is closed AND a closed lid puts it to sleep; 1 otherwise
# (lid open, a desktop, a closed lid kept awake by an external display, or no answer).
# tools/ladder_read_loop.sh asks before it starts a session and while one plays: a
# laptop asleep under its lid still wakes for seconds at a time ("dark wakes"), long
# enough for a launcher to log in and queue, never long enough to play. On 2026-10-09
# that cost a second rated game after the one the lid had closed on. caffeinate does not
# hold a closed lid.
# The power manager's root domain says both things: "AppleClamshellState" (the lid) and
# "AppleClamshellCausesSleep" (No in clamshell mode with a display and power).
state=$(ioreg -r -k AppleClamshellState -d 4 2>/dev/null) || exit 1
printf '%s\n' "$state" | grep -q '"AppleClamshellState" = Yes' || exit 1
printf '%s\n' "$state" | grep -q '"AppleClamshellCausesSleep" = No' && exit 1
exit 0
