# Contributors to EU-Wide Flexibility Needs Assessment Model of the Future European Energy System
#
# Convenience entry point, so that `snakemake -c all` works from the repository
# root. It does nothing but include the model workflow, which locates its own
# configuration through the installed package rather than through
# workflow.basedir, and so behaves identically either way.
#
# Running the model Snakefile directly is equivalent:
#
#     snakemake --snakefile src/euflex/eu_flex_model/Snakefile -c all


include: "src/euflex/eu_flex_model/Snakefile"


# A default_target declared in an included file does not become the default of
# the including workflow, so it is restated here, pointing at the same target.
rule default:
    default_target: True
    input:
        rules.all.input,
