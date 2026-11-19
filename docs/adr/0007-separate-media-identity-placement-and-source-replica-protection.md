# Separate media identity, placement and source replica protection

Status: accepted design; implementation and protocol qualification pending.

A recording media object's stable identity and expected content hash are distinct
from the locations of its retained copies. Transcribe Audio owns a location
catalog and configured storage roots with explicit compatibility for legacy paths.
Moving physical storage must not rewrite recording identity or historical source
provenance. This chooses an application-owned lifecycle over an indefinitely
implicit relationship between a media object and one host pathname.

The initial migration retains exact decoded WAV copies on HDD. Later derivative
reconstruction requires byte-exact qualification from retained sources and decoder
provenance; existing retained-byte validation alone does not qualify eviction and
reconstruction.

A canonical source's last NVMe replica may retire only after the HDD destination
and another durable replica in a separate failure domain are verified. Existing
recording sources can satisfy this with verified identity and durable custody;
path existence alone is insufficient. Missing proof holds that object's old copy
rather than weakening the rule to meet a capacity target. This accepts potentially
slower space recovery to preserve independent source recovery. It does not require
multiple permanent replicas of every disposable derived cache.

Operational grace periods, budgets and quarantine rules are accepted in Plan0097
Q8-Q10. Copy/publication crash recovery, backup rotation, scheduling and rollout
remain pending Q12-Q17; no installed storage state is changed by this decision.

Source: operator acceptance of Plan0097 Q6-Q11 on 2026-11-19.
