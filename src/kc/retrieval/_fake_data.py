"""Default corpus for FakeRetriever.

MOCK DATA ONLY. Invented titles, slugs and texts for tests and local development.
Not real Knowledge Center content. Do not use for evaluation or as a source of facts.
Every chunk_id starts with "fake-" so leaked fake results are easy to spot.
Real sample data will come from Devora's fixtures/ (sample_chunks.jsonl).

The corpus deliberately covers the sensitive points from CLAUDE.md:
- cross-product duplicates ("Expression Samples" in sequence/9 and orchestration/10);
- an exact technical name (Set-CoraSeQuenceSiteAuthentication);
- release notes (to exercise exclude_doc_types);
- an unversioned page (version=None);
- all four products, including version "4" for case-manager and opsmanager.

`score` here is a placeholder (0.0); FakeRetriever computes the real score per query.
"""

from contracts.retrieval import SearchResult

_BASE_URL = "https://knowledgecenter.gcora.genpact.com/help/"

DEFAULT_CORPUS: tuple[SearchResult, ...] = (
    SearchResult(
        chunk_id="fake-seq9-saml-0",
        url=_BASE_URL + "configure-saml-authentication-sequence-9",
        title="Configure SAML Authentication",
        text=(
            "To configure SAML authentication in SeQuence, register the site as a service "
            "provider in your identity provider and upload the SAML metadata file."
        ),
        score=0.0,
        product="sequence",
        version="9",
        doc_type="how-to",
        heading_path=["Administration", "Authentication", "SAML"],
    ),
    SearchResult(
        chunk_id="fake-orch11-saml-0",
        url=_BASE_URL + "saml-single-sign-on-orchestration-11",
        title="SAML Single Sign-On",
        text=(
            "Orchestration supports SAML single sign-on. Configure the identity provider "
            "endpoint and certificate in the authentication settings page."
        ),
        score=0.0,
        product="orchestration",
        version="11",
        doc_type="how-to",
        heading_path=["Administration", "Authentication", "Single Sign-On"],
    ),
    SearchResult(
        chunk_id="fake-seq9-cmdlet-0",
        url=_BASE_URL + "set-corasequencesiteauthentication",
        title="Set-CoraSeQuenceSiteAuthentication",
        text=(
            "The Set-CoraSeQuenceSiteAuthentication cmdlet changes the authentication mode "
            "of a site. Run Set-CoraSeQuenceSiteAuthentication from an elevated PowerShell."
        ),
        score=0.0,
        product="sequence",
        version="9",
        doc_type="reference",
        heading_path=["Reference", "PowerShell Cmdlets"],
    ),
    SearchResult(
        chunk_id="fake-seq9-expr-0",
        url=_BASE_URL + "expression-samples-sequence-9",
        title="Expression Samples",
        text="Expression samples for SeQuence workflows: string, date and lookup expressions.",
        score=0.0,
        product="sequence",
        version="9",
        doc_type="reference",
        heading_path=["Reference", "Expressions", "Expression Samples"],
    ),
    SearchResult(
        chunk_id="fake-orch10-expr-0",
        url=_BASE_URL + "expression-samples-orchestration-10",
        title="Expression Samples",
        text="Expression samples for Orchestration processes: string, date and math expressions.",
        score=0.0,
        product="orchestration",
        version="10",
        doc_type="reference",
        heading_path=["Reference", "Expressions", "Expression Samples"],
    ),
    SearchResult(
        chunk_id="fake-orch10-install-0",
        url=_BASE_URL + "install-orchestration-10",
        title="Install Orchestration",
        text=(
            "Install Orchestration on the application server: run the installer, choose the "
            "database server and enter the service account."
        ),
        score=0.0,
        product="orchestration",
        version="10",
        doc_type="installation",
        heading_path=["Installation", "Install Orchestration"],
    ),
    SearchResult(
        chunk_id="fake-seq8-trouble-0",
        url=_BASE_URL + "workflow-instance-stuck-sequence-8",
        title="Workflow Instance Is Stuck",
        text=(
            "If a workflow instance is stuck, check the workflow service log and restart "
            "the workflow service on the application server."
        ),
        score=0.0,
        product="sequence",
        version="8",
        doc_type="troubleshooting",
        heading_path=["Tips & Troubleshooting", "Workflows"],
    ),
    SearchResult(
        chunk_id="fake-orch11-rn-0",
        url=_BASE_URL + "orchestration-11-release-notes",
        title="Orchestration 11 Release Notes",
        text=(
            "Release notes: new SAML authentication options, improved workflow designer "
            "and fixed issues in the installer."
        ),
        score=0.0,
        product="orchestration",
        version="11",
        doc_type="release-notes",
        heading_path=["Release Notes", "Orchestration 11"],
    ),
    SearchResult(
        chunk_id="fake-cm4-rn-0",
        url=_BASE_URL + "case-manager-4-release-notes",
        title="Case Manager 4 Release Notes",
        text="Release notes: new case dashboard, faster search and fixed issues in case routing.",
        score=0.0,
        product="case-manager",
        version="4",
        doc_type="release-notes",
        heading_path=["Release Notes", "Case Manager 4"],
    ),
    SearchResult(
        chunk_id="fake-orch-overview-0",
        url=_BASE_URL + "orchestration-overview",
        title="Orchestration Overview",
        text=(
            "Orchestration overview: model processes, automate workflows and connect "
            "systems in one platform."
        ),
        score=0.0,
        product="orchestration",
        version=None,
        doc_type="overview",
        heading_path=["Overview"],
    ),
    SearchResult(
        chunk_id="fake-ops4-howto-0",
        url=_BASE_URL + "create-case-type-opsmanager-4",
        title="Create a Case Type",
        text="To create a case type in OpsManager, open the case designer and define case fields.",
        score=0.0,
        product="opsmanager",
        version="4",
        doc_type="how-to",
        heading_path=["Administration", "Case Types"],
    ),
    SearchResult(
        chunk_id="fake-seq7-upgrade-0",
        url=_BASE_URL + "upgrade-sequence-7",
        title="Upgrade SeQuence",
        text=(
            "Upgrade SeQuence: back up the database, run the upgrade installer and verify "
            "the site after the upgrade."
        ),
        score=0.0,
        product="sequence",
        version="7",
        doc_type="installation",
        heading_path=["Installation", "Upgrade"],
    ),
)
