# Derive affiliations from durable role appointments

An affiliation is a deterministic read-model grouping keyed by person and
organization, while each role appointment remains the independently
reviewable, temporal ledger assertion identified by `role_id`. We chose this
over a second mutable affiliation authority because provider strings may show
membership without a known role, several simultaneous roles must coexist, and
correction or reversal must affect one assertion without rewriting its peers.
The compact primary affiliation is therefore a ranked display projection, not
an employer field or accepted fact.
