# Use durable conversation identities

Conversations receive opaque durable IDs that survive artifact renames,
regeneration, and migration into user-scoped central storage. Paths and content
hashes remain useful aliases and integrity evidence, but neither is stable
enough to own identity: paths change when artifacts move, while content hashes
change when transcripts are corrected. New IDs are assigned during normalized
transcript creation, while existing artifacts are backfilled lazily during
processing or migration. Recording identity remains separate: one recording
per conversation is the default, but multiple recording segments may later be
associated with the same conversation without changing their recording IDs.
