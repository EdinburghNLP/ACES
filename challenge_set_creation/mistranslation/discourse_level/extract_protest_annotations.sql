SELECT
ann.id, ann.candidate, ann.annotator_id, ann.ant_annotation, ann.anaph_annotation, ann.remarks,
pro.srccorpus, pro.tgtcorpus, pro.example_no, pro.srcpro, pro.srcpos, pro.srcproindex, pro.line,
tran.tgtpos as trans_tgtpos,
ref_tran.tgtpos as ref_tgtpos,
ref_tran.ant_no,
cat.description,
source_s.sentence as source,
target_s.sentence as mt_translation,
ref.sentence as reference
FROM annotations AS ann
LEFT JOIN pro_candidates AS pro
ON ann.candidate = pro.id
LEFT JOIN pro_antecedents as ant
ON pro.srccorpus = ant.srccorpus AND pro.tgtcorpus = ant.tgtcorpus AND pro.example_no = ant.example_no
LEFT JOIN translations as tran
ON pro.tgtcorpus = tran.tgtcorpus AND pro.line = tran.line AND pro.example_no = tran.example_no
LEFT JOIN translations as ref_tran
ON pro.line = ref_tran.line AND pro.example_no = ref_tran.example_no
LEFT JOIN sentences as source_s
ON pro.line = source_s.line AND pro.srccorpus = source_s.corpus
LEFT JOIN sentences as target_s
ON pro.line = target_s.line AND pro.tgtcorpus = target_s.corpus
LEFT JOIN sentences as ref
ON pro.line = ref.line
LEFT JOIN categories AS cat
ON pro.category_no = cat.id
WHERE (
cat.description = 'pleonastic_it'
OR cat.description = 'anaphoric_intra_subject_it'
OR cat.description = 'anaphoric_intra_non-subject_it'
OR cat.description = 'anaphoric_intra_they'
OR (cat.description = 'anaphoric_singular_they' AND pro.line = ant.line)
OR (cat.description = 'anaphoric_group_it-they' AND pro.line = ant.line)
)
AND pro.srccorpus > 1 AND pro.tgtcorpus > 2
AND ref.corpus = 2
AND ref_tran.tgtcorpus = 2
AND ref_tran.ant_no IS NULL
AND tran.ant_no IS NULL;

