# Papers I need, and what each one is holding up

Generated from the library's own citations. Every value below is stored
as MEASURED with the PMID beside it, and for these the full text could
not be reached -- so the number came from an abstract, with no table and
no method behind it.

A PDF in `docs/papers/` named `PMID.pdf` is all I need; I parse them with
pypdf. No credentials, no proxy, nothing shared.

| PMID | DOI | What I need from it |
|---|---|---|
| 11404363 | 10.1074/jbc.m104844200 | Values behind: bond_lifetime_seconds, kd_nm, koff_per_s, kon_per_nm_per_s |
| 9422729 | 10.1074/jbc.273.2.763 | Values behind: bond_lifetime_seconds, kd_nm, koff_per_s, kon_per_nm_per_s |
| 9780171 | 10.4049/jimmunol.161.8.4016 | Values behind: bond_lifetime_seconds, kd_nm, koff_per_s, kon_per_nm_per_s |
| 9829984 | 10.1074/jbc.273.49.32506 | Values behind: bond_lifetime_seconds, kd_nm, koff_per_s, kon_per_nm_per_s |
| 20178337 | 10.1021/bi100022c | Values behind: kcat_per_s, km_um |
| 27647916 | 10.1073/pnas.1605399113 | Fig. S2 legend (density 3 +/- 0.5 um^-2 by TIRF vs 4.5 by SEM) and the microvillar diameter measurements. We have these from the abstract only. |
| 10381624 | 10.1038/sj.cdd.4400497 | Values behind: kcat_per_s |
| 12451422 | 10.1007/s00249-002-0248-9 | Values behind: kd_nm |
| 15130947 | 10.1182/blood-2004-02-0437 | Table/figure giving microvillar LENGTH per lymphocyte source, and the 2-4 per um^2 DENSITY range. We store the median 0.35 um from the abstract; the per-source table is what separates T from B and resting from activated. |
| 15209560 | 10.1042/bj20040809 | Values behind: kcat_per_s |
| 15771445 | 10.1021/jm040842z | Values behind: kd_nm |
| 16490769 | 10.1177/1087057105285048 | Values behind: km_um |
| 16840338 | 10.1128/jvi.01623-05 | Values behind: replication_cycle_hours |
| 2272970 | 10.1152/jappl.1990.69.5.1767 | Leukocyte diameters by subtype. Holds up radius_um. |
| 25679760 | 10.1016/j.cell.2015.01.007 | Amoeboid migration speeds under confinement. Holds up migration_speed_um_per_min. |

**15 papers.** Priority order: 15130947 and 27647916 next (every
microvillus number), then 308396 (the nuclear fraction).

## Done

**35676539** opened on 2026-10-21, the day the Unpaywall address was
added -- and then only because the route was fixed to READ the PDF it
fetches instead of noting that one exists. It confirms the stored value:
the paper says `kcat of (18 +/- 4) s-1` and TMPRSS2 stores 18.0. It also
reports `Km = (200 +/- 80) uM` on Boc-QAR-AMC, which is not stored and
could be -- noting that the substrate is synthetic, so it is not a
physiological Km.

Unpaywall opened exactly one of sixteen. Eleven report a bronze, hybrid
or green licence with no fetchable copy, which means a copy is legally
free to read and the publisher blocks automated access anyway; four are
not open access at all. An OA flag describes a licence, not a door.

One lead that looked promising and was not: Unpaywall gives a PMC id for
27647916 that Europe PMC never returned (PMC5056101), but NCBI efetch
serves metadata and an abstract for it with no article body -- PNAS has
not deposited the full text for redistribution. The code's existing
judgement was right.


**6775712** arrived 2026-10-20 and Table 2 is read. All four stored
granule fractions match it exactly -- 23.0 / 15.4 / 2.6 / 0.2 -- so the
BioNumbers transcription was faithful. What the BNIDs did not carry is
the paper's Appendix: the four are overestimates, by up to 19% / 28% /
29% / 44%, because a granule's diameter is comparable to the 0.08 um
section and it gets counted too often. That bias is now recorded beside
each value. The same Appendix clears the nuclear fraction explicitly.

Reading it also found a reader bug: pypdf refuses any page whose font
descriptor declares two embedded programs, and every page of a
ScienceDirect PDF does. All ten pages extracted to nothing until that
was fixed, so the next publisher PDF would have failed the same way.

Everything else the library cites is already readable in full.
