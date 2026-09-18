# Added-Information Case Studies

## mostly_supported: Re3Syn-generated data

- Paper: `ACL:2025.acl-long.1518::dataset::0::re3syn-generated-data`
- Condition: `oracle`
- Added-information score: 0.000

| Query ACU | Status | Best Prior ACU | Delta Type | Rationale |
| --- | --- | --- | --- | --- |
| RE3SYN uses dependency recognition instead of just similarity to synthesize long-context data. | not_comparable |  | other | The prior ACUs are about LongBench and LongBench-E (benchmark scope, task categories, test instances, formatting, and length distribution), not about RE3SYN or any synthesis method. Therefore they do not provide evidence for or against the claim that RE3SYN uses dependency recognition instead of similarity. |
| The framework uses OPT-350m to calculate perplexity for dependency recognition. | not_comparable |  | other | None of the prior ACUs mention OPT-350m, perplexity, dependency recognition, or RE3SYN’s framework details. The supplied prior support is unrelated LongBench metadata, so this claim cannot be compared against it. |
| The framework reduces document redundancy compared to similarity-based concatenation. | not_comparable |  | other | The prior ACUs only describe LongBench and LongBench-E benchmark properties and do not address document redundancy or similarity-based concatenation. As a result, there is no basis in the supplied prior ACUs to support or contradict this RE3SYN claim. |

## extension: ReDial

- Paper: `ACL:2025.acl-long.317::dataset::0::redial`
- Condition: `oracle`
- Added-information score: 1.000

| Query ACU | Status | Best Prior ACU | Delta Type | Rationale |
| --- | --- | --- | --- | --- |
| ReDial is the first high-quality, end-to-end human-annotated SE-AAVE parallel reasoning benchmark. | unsupported |  | task/domain | The prior ACUs discuss GSM8K, HumanEval, and FOLIO, not ReDial or any SE-AAVE parallel reasoning benchmark. No prior ACU supports the claim that ReDial is the first such benchmark or that it is end-to-end human-annotated. |
| The dataset contains over 1.2K parallel query pairs covering four reasoning categories. | unsupported |  | scale/coverage | None of the prior ACUs mention ReDial, parallel query pairs, or four reasoning categories. The size and category coverage claim is not supported by the supplied prior ACUs. |
| Annotators were hired to rewrite SE queries into AAVE while preserving critical information. | unsupported |  | annotation/protocol | The prior ACUs do not describe any annotation workflow for rewriting SE queries into AAVE, nor do they mention preserving critical information during annotation. This query claim is unsupported by the supplied prior evidence. |
| Quality control involved cross-checking by AAVE speakers and sanity checks by non-AAVE speakers and GPT-4o. | unsupported |  | availability/quality | The prior ACUs contain no information about quality control, AAVE speakers, non-AAVE speakers, or GPT-4o. Therefore the specific QC procedure in the query ACU is unsupported. |

## high_added_information: MultiAgentBench

- Paper: `ACL:2025.acl-long.421::dataset::0::multiagentbench`
- Condition: `oracle`
- Added-information score: 1.000

| Query ACU | Status | Best Prior ACU | Delta Type | Rationale |
| --- | --- | --- | --- | --- |
| MultiAgentBench covers six diverse interactive scenarios including research, Minecraft, database error analysis, coding, Werewolf, and Bargaining. | unsupported |  | task/domain | The prior ACUs describe AGENTBENCH as 8 distinct environments and categorize them as Code, Game, and Web groundings, but they do not support MultiAgentBench having six specific scenarios or list research, Minecraft, database error analysis, coding, Werewolf, and Bargaining. |
| The benchmark uses milestone-based KPIs to measure task completion and individual agent contributions. | unsupported |  | annotation/protocol | None of the prior ACUs mention milestone-based KPIs, task completion metrics, or measuring individual agent contributions; they only describe AGENTBENCH's environments, model evaluation, and a failure mode. |
| The framework supports four coordination protocols: star, chain, tree, and graph. | unsupported |  | annotation/protocol | The prior ACUs do not mention any coordination protocols or communication structures, so there is no support for star, chain, tree, and graph being supported. |
| The benchmark includes both collaborative and competitive scenarios. | unsupported |  | task/domain | The prior ACUs say AGENTBENCH has environments in Code, Game, and Web groundings, but they do not indicate whether the benchmark includes collaborative, competitive, or both types of scenarios. |
