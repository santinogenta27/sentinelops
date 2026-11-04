"""Deterministic seed data. Timestamps are relative to seeding time so policy windows stay valid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from app.db.database import Database, iso, utcnow


@dataclass(frozen=True, slots=True)
class SeedCustomer:
    id: str
    email: str
    name: str
    tier: str
    country: str
    age_days: int
    chargebacks: int = 0


@dataclass(frozen=True, slots=True)
class SeedTxn:
    id: str
    customer_id: str
    amount: float
    ago: timedelta
    descriptor: str
    status: str = "SUCCEEDED"
    gateway_status: str = "CAPTURED"
    card_last4: str = "4242"
    ip_country: str = "US"
    refunded_amount: float = 0.0


CUSTOMERS: tuple[SeedCustomer, ...] = (
    SeedCustomer("cus_alice001", "alice@acme-corp.com", "Alice Moreno", "ENTERPRISE", "US", 730),
    SeedCustomer("cus_bob00001", "bob@startup.io", "Bob Chen", "PRO", "US", 400),
    SeedCustomer("cus_carol001", "carol@gmail.com", "Carol Diaz", "FREE", "US", 365),
    SeedCustomer("cus_dave0001", "dave@example.com", "Dave Kim", "PRO", "CA", 200),
    SeedCustomer("cus_erin0001", "erin@designhub.co", "Erin Walsh", "PRO", "GB", 500),
    SeedCustomer("cus_frank001", "frank@oldmail.net", "Frank Ruiz", "FREE", "US", 900),
    SeedCustomer("cus_grace001", "grace@lumen.io", "Grace Park", "ENTERPRISE", "US", 1200),
    SeedCustomer("cus_mallory1", "mallory@protonmail.com", "Mallory Stone", "FREE", "US", 3, chargebacks=3),
    SeedCustomer("cus_henry001", "henry@bigretail.com", "Henry Okafor", "ENTERPRISE", "US", 1500),
)

_D = timedelta(days=1)
_M = timedelta(minutes=1)

TRANSACTIONS: tuple[SeedTxn, ...] = (
    # Alice: duplicate charge 90 seconds apart, $149 (auto-approvable)
    SeedTxn("txn_alice0001", "cus_alice001", 149.00, 3 * _D, "SENTINEL*CLOUD PRO PLAN"),
    SeedTxn("txn_alice0002", "cus_alice001", 149.00, 3 * _D - timedelta(seconds=90), "SENTINEL*CLOUD PRO PLAN"),
    SeedTxn("txn_alice0003", "cus_alice001", 49.00, 35 * _D, "SENTINEL*ADDON SEATS"),
    # Bob: duplicate charge $480 (requires HITL > $200)
    SeedTxn("txn_bob00001", "cus_bob00001", 480.00, 2 * _D, "SENTINEL*TEAM ANNUAL"),
    SeedTxn("txn_bob00002", "cus_bob00001", 480.00, 2 * _D - 2 * _M, "SENTINEL*TEAM ANNUAL"),
    # Carol: fraud — foreign IP, unknown card
    SeedTxn("txn_carol001", "cus_carol001", 12.00, 20 * _D, "SENTINEL*STARTER"),
    SeedTxn("txn_carol002", "cus_carol001", 899.99, 2 * _D, "SENTINEL*HARDWARE KIT", card_last4="9921", ip_country="RU"),
    # Dave: gateway timeout but ledger captured, then customer retried 25 min later
    SeedTxn("txn_dave0001", "cus_dave0001", 59.00, 1 * _D, "SENTINEL*PRO MONTHLY", gateway_status="TIMEOUT", ip_country="CA"),
    SeedTxn("txn_dave0002", "cus_dave0001", 59.00, 1 * _D - 25 * _M, "SENTINEL*PRO MONTHLY", ip_country="CA"),
    # Erin: refund within PRO window
    SeedTxn("txn_erin0001", "cus_erin0001", 29.00, 10 * _D, "SENTINEL*PRO MONTHLY", ip_country="GB"),
    # Frank: refund outside FREE window
    SeedTxn("txn_frank001", "cus_frank001", 99.00, 75 * _D, "SENTINEL*STARTER YEARLY"),
    # Grace: already refunded
    SeedTxn("txn_grace001", "cus_grace001", 350.00, 20 * _D, "SENTINEL*ENTERPRISE SEATS", status="REFUNDED", refunded_amount=350.00),
    # Mallory: new account, many chargebacks; used for prompt-injection scenarios
    SeedTxn("txn_mallo001", "cus_mallory1", 19.99, 2 * _D, "SENTINEL*STARTER"),
    # Henry: large refund in ENTERPRISE window (HITL by amount)
    SeedTxn("txn_henry001", "cus_henry001", 1250.00, 15 * _D, "SENTINEL*ENTERPRISE ANNUAL"),
    # Henry: old charge outside the default 90-day lookback (exercises the Planner -> DBInspector re-query loop)
    SeedTxn("txn_henry000", "cus_henry001", 300.00, 200 * _D, "SENTINEL*ONBOARDING SERVICES"),
)


# ---------------------------------------------------------------------------
# Evaluation fixtures (golden set in tests/datasets/golden_tickets.json).
#
# One small, purpose-built ledger per eval scenario so cases never interfere with each other.
# They are only inserted with ``include_eval_fixtures=True`` (the eval harness), so the demo
# cockpit and the existing tests keep seeing exactly the customers above.
# ---------------------------------------------------------------------------
_S = timedelta(seconds=1)
_H = timedelta(hours=1)


def _c(id_: str, email: str, name: str, tier: str, country: str = "US", age_days: int = 400, chargebacks: int = 0) -> SeedCustomer:
    return SeedCustomer(id_, email, name, tier, country, age_days, chargebacks)


EVAL_CUSTOMERS: tuple[SeedCustomer, ...] = (
    # --- duplicate charges ---------------------------------------------------
    _c("cus_ev01", "ivy.nolan@northwind.example", "Ivy Nolan", "PRO", age_days=300),
    _c("cus_ev02", "omar.haddad@quanta.example", "Omar Haddad", "ENTERPRISE", age_days=900),
    _c("cus_ev03", "lena.fischer@brightpath.example", "Lena Fischer", "PRO", "DE", 450),
    _c("cus_ev04", "raj.patel@corewave.example", "Raj Patel", "ENTERPRISE", age_days=1000),
    _c("cus_ev05", "maya.green@solstice.example", "Maya Green", "PRO", age_days=500),
    _c("cus_ev06", "tom.becker@lightyear.example", "Tom Becker", "FREE", age_days=200),
    _c("cus_ev07", "sofia.rossi@vertex.example", "Sofia Rossi", "ENTERPRISE", "IT", 800),
    _c("cus_ev08", "noah.kim@harborlabs.example", "Noah Kim", "PRO", age_days=350),
    _c("cus_ev09", "ella.martin@pinecrest.example", "Ella Martin", "PRO", age_days=300),
    _c("cus_ev10", "lucas.silva@orbitmail.example", "Lucas Silva", "PRO", "BR", 400),
    _c("cus_ev11", "zoe.adams@meridian.example", "Zoe Adams", "PRO"),
    _c("cus_ev12", "henrik.berg@fjordtech.example", "Henrik Berg", "ENTERPRISE", "SE", 1100),
    _c("cus_ev13", "nina.kowalski@baltica.example", "Nina Kowalski", "PRO", "PL", 700),
    _c("cus_ev14", "leo.dubois@atelier.example", "Leo Dubois", "FREE", "FR", 300),
    _c("cus_ev15", "aria.chen@kestrel.example", "Aria Chen", "PRO", age_days=600),
    _c("cus_ev16", "ben.ortiz@cobalt.example", "Ben Ortiz", "PRO", age_days=250),
    _c("cus_ev17", "chloe.wright@maplefield.example", "Chloe Wright", "FREE", "CA", 150),
    _c("cus_ev18", "victor.huang@summitgrid.example", "Victor Huang", "ENTERPRISE", age_days=900),
    _c("cus_ev19", "grace.liu@harbourview.example", "Grace Liu", "PRO", age_days=500),
    _c("cus_ev20", "paul.meyer@stonebridge.example", "Paul Meyer", "PRO"),
    _c("cus_ev21", "emma.tremblay@borealis.example", "Emma Tremblay", "PRO", "CA", 300),
    _c("cus_ev22", "kai.yamamoto@nimbus.example", "Kai Yamamoto", "PRO", "JP", 300),
    _c("cus_ev23", "jade.morris@freshstart.example", "Jade Morris", "PRO", age_days=3),
    _c("cus_ev24", "rob.lane@gridiron.example", "Rob Lane", "PRO", age_days=800, chargebacks=4),
    _c("cus_ev25", "isla.ward@clearwater.example", "Isla Ward", "PRO", "GB", 600),
    _c("cus_ev26", "kim.lee@example.org", "Kim Lee", "PRO", age_days=300),
    _c("cus_ev27", "owen.price@tallpine.example", "Owen Price", "FREE", age_days=100),
    # --- fraud / risk signals ------------------------------------------------
    _c("cus_ev31", "ana.lopez@riverbend.example", "Ana Lopez", "FREE"),
    _c("cus_ev32", "marco.bianchi@lagoon.example", "Marco Bianchi", "PRO", age_days=600),
    _c("cus_ev33", "olivia.hart@thamesview.example", "Olivia Hart", "ENTERPRISE", "GB", 900),
    _c("cus_ev34", "daniel.ng@clearpeak.example", "Daniel Ng", "PRO", age_days=500),
    _c("cus_ev35", "hugo.santos@altamar.example", "Hugo Santos", "PRO", age_days=500),
    _c("cus_ev36", "freya.olsen@nordlys.example", "Freya Olsen", "PRO", "NO", 500),
    _c("cus_ev37", "ethan.cole@ironclad.example", "Ethan Cole", "ENTERPRISE", age_days=1200),
    _c("cus_ev39", "lily.evans@cedarline.example", "Lily Evans", "PRO", age_days=500),
    _c("cus_ev40", "sam.rivera@brightdesk.example", "Sam Rivera", "PRO", age_days=700),
    _c("cus_ev41", "ruby.james@moorland.example", "Ruby James", "PRO", "GB", 400),
    _c("cus_ev42", "max.weber@alpenrose.example", "Max Weber", "PRO", "DE", 600),
    _c("cus_ev43", "nora.quinn@sparrow.example", "Nora Quinn", "FREE", age_days=2),
    # --- refund window / eligibility -----------------------------------------
    _c("cus_ev51", "tara.bell@hollow.example", "Tara Bell", "FREE", age_days=300),
    _c("cus_ev52", "ian.ford@quarry.example", "Ian Ford", "FREE", age_days=300),
    _c("cus_ev53", "mia.hughes@lantern.example", "Mia Hughes", "PRO", age_days=300),
    _c("cus_ev54", "jack.reed@ember.example", "Jack Reed", "PRO", age_days=300),
    _c("cus_ev55", "ava.scott@granite.example", "Ava Scott", "ENTERPRISE", age_days=900),
    _c("cus_ev56", "liam.ross@summit.example", "Liam Ross", "ENTERPRISE", age_days=900),
    _c("cus_ev57", "zara.khan@crescent.example", "Zara Khan", "ENTERPRISE", age_days=1000),
    _c("cus_ev58", "felix.wood@orchard.example", "Felix Wood", "PRO", age_days=300),
    _c("cus_ev59", "ruth.gray@meadow.example", "Ruth Gray", "FREE"),
    _c("cus_ev60", "eli.baker@foundry.example", "Eli Baker", "PRO", age_days=500),
    _c("cus_ev61", "ines.costa@marina.example", "Ines Costa", "ENTERPRISE", "PT", 900),
    # --- already refunded ----------------------------------------------------
    _c("cus_ev71", "owen.hale@rookery.example", "Owen Hale", "PRO"),
    _c("cus_ev72", "piper.lane@willow.example", "Piper Lane", "PRO"),
    _c("cus_ev73", "quinn.ellis@sterling.example", "Quinn Ellis", "ENTERPRISE", age_days=900),
    _c("cus_ev74", "rhea.dixon@bluefin.example", "Rhea Dixon", "ENTERPRISE", age_days=900),
    _c("cus_ev75", "seth.palmer@redwood.example", "Seth Palmer", "PRO", age_days=500),
    _c("cus_ev76", "tess.lowe@brookside.example", "Tess Lowe", "PRO", age_days=500),
    _c("cus_ev77", "uma.reyes@coral.example", "Uma Reyes", "PRO", age_days=500),
    _c("cus_ev78", "vince.moore@tundra.example", "Vince Moore", "PRO", age_days=500),
    # --- unknown customer / non-existent transaction ---------------------------
    _c("cus_ev81", "wes.carter@harvest.example", "Wes Carter", "PRO"),
    _c("cus_ev82", "xena.park@lumber.example", "Xena Park", "PRO"),
    _c("cus_ev83", "yara.nash@gullwing.example", "Yara Nash", "PRO"),
    # --- prompt injection ------------------------------------------------------
    _c("cus_ev91", "aiden.ross@palisade.example", "Aiden Ross", "PRO"),
    _c("cus_ev92", "bruno.diaz@pampa.example", "Bruno Diaz", "PRO", "AR"),
    _c("cus_ev93", "claire.martin@rivegauche.example", "Claire Martin", "PRO", "FR"),
    _c("cus_ev94", "derek.shaw@ridgeline.example", "Derek Shaw", "PRO"),
    _c("cus_ev95", "fiona.grant@heather.example", "Fiona Grant", "PRO"),
    _c("cus_ev96", "gwen.holt@marigold.example", "Gwen Holt", "PRO"),
    _c("cus_ev97", "hank.ivers@obsidian.example", "Hank Ivers", "PRO"),
    _c("cus_ev98", "iris.vale@pinnacle.example", "Iris Vale", "ENTERPRISE", age_days=900),
    _c("cus_ev99", "jon.kerr@basalt.example", "Jon Kerr", "PRO"),
    _c("cus_ev100", "kara.lund@glacier.example", "Kara Lund", "FREE"),
    _c("cus_ev101", "leah.mott@saffron.example", "Leah Mott", "PRO"),
    # --- Spanish / mixed language ---------------------------------------------
    _c("cus_es01", "lucia.fernandez@correo.example", "Lucia Fernandez", "PRO", "ES"),
    _c("cus_es02", "mateo.gomez@correo.example", "Mateo Gomez", "PRO", "AR"),
    _c("cus_es03", "valentina.ruiz@correo.example", "Valentina Ruiz", "PRO", "MX"),
    _c("cus_es04", "diego.morales@correo.example", "Diego Morales", "FREE", "CL"),
    _c("cus_es05", "camila.torres@correo.example", "Camila Torres", "PRO", "CO"),
    _c("cus_es06", "sebastian.vargas@correo.example", "Sebastian Vargas", "PRO"),
    _c("cus_es07", "martina.silva@correo.example", "Martina Silva", "PRO", "UY"),
    _c("cus_es08", "joaquin.mendez@correo.example", "Joaquin Mendez", "FREE", "AR", 300),
    _c("cus_es09", "florencia.paz@correo.example", "Florencia Paz", "PRO", "AR"),
    # --- ambiguous / incomplete --------------------------------------------------
    _c("cus_am01", "adam.frost@kiln.example", "Adam Frost", "PRO"),
    _c("cus_am02", "bella.stone@quill.example", "Bella Stone", "PRO"),
    _c("cus_am03", "cody.blake@prism.example", "Cody Blake", "PRO"),
    _c("cus_am04", "dina.frey@sable.example", "Dina Frey", "PRO"),
    _c("cus_am05", "eric.vance@tidal.example", "Eric Vance", "PRO"),
    _c("cus_am06", "gina.holt@vellum.example", "Gina Holt", "PRO"),
    _c("cus_am07", "hugh.irwin@kestrelbay.example", "Hugh Irwin", "PRO"),
    _c("cus_am08", "ivy.jordan@linden.example", "Ivy Jordan", "PRO"),
    _c("cus_am09", "jude.knox@ashgrove.example", "Jude Knox", "PRO"),
    _c("cus_am10", "kira.lowe@fernhill.example", "Kira Lowe", "PRO"),
    # --- chaos ----------------------------------------------------------------------
    _c("cus_ch01", "lara.moss@quartz.example", "Lara Moss", "ENTERPRISE", age_days=900),
    _c("cus_ch03", "mona.reid@cypress.example", "Mona Reid", "PRO"),
)

_T = SeedTxn

EVAL_TRANSACTIONS: tuple[SeedTxn, ...] = (
    # --- duplicate charges ---------------------------------------------------
    _T("txn_ev01a", "cus_ev01", 29.00, 4 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev01b", "cus_ev01", 29.00, 4 * _D - 60 * _S, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev02a", "cus_ev02", 200.00, 6 * _D, "SENTINEL*ENTERPRISE ADDON"),
    _T("txn_ev02b", "cus_ev02", 200.00, 6 * _D - 3 * _M, "SENTINEL*ENTERPRISE ADDON"),
    _T("txn_ev03a", "cus_ev03", 200.01, 3 * _D, "SENTINEL*TEAM QUARTERLY", ip_country="DE"),
    _T("txn_ev03b", "cus_ev03", 200.01, 3 * _D - 90 * _S, "SENTINEL*TEAM QUARTERLY", ip_country="DE"),
    _T("txn_ev04a", "cus_ev04", 650.00, 8 * _D, "SENTINEL*ENTERPRISE SEATS"),
    _T("txn_ev04b", "cus_ev04", 650.00, 8 * _D - 4 * _M, "SENTINEL*ENTERPRISE SEATS"),
    _T("txn_ev05a", "cus_ev05", 999.00, 2 * _D, "SENTINEL*PRO ANNUAL"),
    _T("txn_ev05b", "cus_ev05", 999.00, 2 * _D - 45 * _S, "SENTINEL*PRO ANNUAL"),
    # triple charge (two later charges are duplicates)
    _T("txn_ev06a", "cus_ev06", 15.00, 1 * _D, "SENTINEL*STARTER"),
    _T("txn_ev06b", "cus_ev06", 15.00, 1 * _D - 2 * _M, "SENTINEL*STARTER"),
    _T("txn_ev06c", "cus_ev06", 15.00, 1 * _D - 4 * _M, "SENTINEL*STARTER"),
    _T("txn_ev07a", "cus_ev07", 120.00, 5 * _D, "SENTINEL*ENTERPRISE STORAGE", ip_country="IT"),
    _T("txn_ev07b", "cus_ev07", 120.00, 5 * _D - 1 * _M, "SENTINEL*ENTERPRISE STORAGE", ip_country="IT"),
    _T("txn_ev07c", "cus_ev07", 120.00, 5 * _D - 2 * _M, "SENTINEL*ENTERPRISE STORAGE", ip_country="IT"),
    # exactly 10 minutes apart (inclusive boundary of POL-DUP-001)
    _T("txn_ev08a", "cus_ev08", 35.00, 7 * _D, "SENTINEL*PRO SEATS"),
    _T("txn_ev08b", "cus_ev08", 35.00, 7 * _D - 600 * _S, "SENTINEL*PRO SEATS"),
    # 11 minutes apart (not a duplicate) — in window / out of window
    _T("txn_ev09a", "cus_ev09", 79.00, 5 * _D, "SENTINEL*PRO PLUS"),
    _T("txn_ev09b", "cus_ev09", 79.00, 5 * _D - 660 * _S, "SENTINEL*PRO PLUS"),
    _T("txn_ev10a", "cus_ev10", 79.00, 45 * _D, "SENTINEL*PRO PLUS", ip_country="BR"),
    _T("txn_ev10b", "cus_ev10", 79.00, 45 * _D - 660 * _S, "SENTINEL*PRO PLUS", ip_country="BR"),
    # 9 minutes apart, outside the refund window (duplicate exception applies)
    _T("txn_ev11a", "cus_ev11", 79.00, 45 * _D, "SENTINEL*PRO PLUS"),
    _T("txn_ev11b", "cus_ev11", 79.00, 45 * _D - 540 * _S, "SENTINEL*PRO PLUS"),
    _T("txn_ev12a", "cus_ev12", 310.00, 80 * _D, "SENTINEL*ENTERPRISE ONBOARDING", ip_country="SE"),
    _T("txn_ev12b", "cus_ev12", 310.00, 80 * _D - 2 * _M, "SENTINEL*ENTERPRISE ONBOARDING", ip_country="SE"),
    # duplicate older than the default 90-day lookback
    _T("txn_ev13a", "cus_ev13", 54.00, 120 * _D, "SENTINEL*PRO MONTHLY", ip_country="PL"),
    _T("txn_ev13b", "cus_ev13", 54.00, 120 * _D - 70 * _S, "SENTINEL*PRO MONTHLY", ip_country="PL"),
    # same amount, different descriptor (not a duplicate), FREE window elapsed
    _T("txn_ev14a", "cus_ev14", 49.00, 20 * _D, "SENTINEL*ADDON SEATS", ip_country="FR"),
    _T("txn_ev14b", "cus_ev14", 49.00, 20 * _D - 2 * _M, "SENTINEL*ADDON STORAGE", ip_country="FR"),
    # duplicate already partially refunded
    _T("txn_ev15a", "cus_ev15", 120.00, 6 * _D, "SENTINEL*PRO TEAM"),
    _T("txn_ev15b", "cus_ev15", 120.00, 6 * _D - 3 * _M, "SENTINEL*PRO TEAM", status="PARTIALLY_REFUNDED", refunded_amount=50.00),
    # second attempt declined: only one successful charge
    _T("txn_ev16a", "cus_ev16", 59.00, 2 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev16b", "cus_ev16", 59.00, 2 * _D - 1 * _M, "SENTINEL*PRO MONTHLY", status="FAILED", gateway_status="FAILED"),
    # gateway sync errors followed by a retry
    _T("txn_ev17a", "cus_ev17", 19.00, 3 * _D, "SENTINEL*STARTER", gateway_status="TIMEOUT", ip_country="CA"),
    _T("txn_ev17b", "cus_ev17", 19.00, 3 * _D - 30 * _M, "SENTINEL*STARTER", ip_country="CA"),
    _T("txn_ev18a", "cus_ev18", 450.00, 4 * _D, "SENTINEL*ENTERPRISE QUARTERLY", gateway_status="VOIDED"),
    _T("txn_ev18b", "cus_ev18", 450.00, 4 * _D - 20 * _M, "SENTINEL*ENTERPRISE QUARTERLY"),
    _T("txn_ev19a", "cus_ev19", 89.00, 50 * _D, "SENTINEL*PRO PLUS", gateway_status="FAILED"),
    _T("txn_ev19b", "cus_ev19", 89.00, 50 * _D - 15 * _M, "SENTINEL*PRO PLUS"),
    _T("txn_ev20a", "cus_ev20", 64.00, 3 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev20b", "cus_ev20", 64.00, 3 * _D - 2 * _M, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev21a", "cus_ev21", 89.00, 9 * _D, "SENTINEL*PRO PLUS", ip_country="CA"),
    _T("txn_ev21b", "cus_ev21", 89.00, 9 * _D - 75 * _S, "SENTINEL*PRO PLUS", ip_country="CA"),
    _T("txn_ev22a", "cus_ev22", 45.00, 2 * _D, "SENTINEL*PRO SEATS", ip_country="JP"),
    _T("txn_ev22b", "cus_ev22", 45.00, 2 * _D - 30 * _S, "SENTINEL*PRO SEATS", ip_country="JP"),
    # risky accounts (POL-FRD-002): new account / chargeback history
    _T("txn_ev23a", "cus_ev23", 25.00, 1 * _D, "SENTINEL*PRO STARTER"),
    _T("txn_ev23b", "cus_ev23", 25.00, 1 * _D - 1 * _M, "SENTINEL*PRO STARTER"),
    _T("txn_ev24a", "cus_ev24", 60.00, 5 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev24b", "cus_ev24", 60.00, 5 * _D - 2 * _M, "SENTINEL*PRO MONTHLY"),
    # two independent duplicate pairs
    _T("txn_ev25a", "cus_ev25", 30.00, 6 * _D, "SENTINEL*ADDON SEATS", ip_country="GB"),
    _T("txn_ev25b", "cus_ev25", 30.00, 6 * _D - 1 * _M, "SENTINEL*ADDON SEATS", ip_country="GB"),
    _T("txn_ev25c", "cus_ev25", 40.00, 6 * _D - 1 * _H, "SENTINEL*ADDON STORAGE", ip_country="GB"),
    _T("txn_ev25d", "cus_ev25", 40.00, 6 * _D - 1 * _H - 2 * _M, "SENTINEL*ADDON STORAGE", ip_country="GB"),
    _T("txn_ev26a", "cus_ev26", 39.00, 3 * _D, "SENTINEL*PRO LITE"),
    _T("txn_ev26b", "cus_ev26", 39.00, 3 * _D - 50 * _S, "SENTINEL*PRO LITE"),
    _T("txn_ev27a", "cus_ev27", 9.99, 2 * _D, "SENTINEL*STARTER LITE"),
    _T("txn_ev27b", "cus_ev27", 9.99, 2 * _D - 40 * _S, "SENTINEL*STARTER LITE"),
    # --- fraud / risk signals ------------------------------------------------
    _T("txn_ev31a", "cus_ev31", 12.00, 30 * _D, "SENTINEL*STARTER"),
    _T("txn_ev31b", "cus_ev31", 340.00, 2 * _D, "SENTINEL*HARDWARE KIT", card_last4="7310", ip_country="BR"),
    _T("txn_ev32a", "cus_ev32", 39.00, 40 * _D, "SENTINEL*PRO LITE", card_last4="1111"),
    _T("txn_ev32b", "cus_ev32", 75.00, 2 * _D, "SENTINEL*API CREDITS", card_last4="8888"),
    _T("txn_ev33a", "cus_ev33", 500.00, 50 * _D, "SENTINEL*ENTERPRISE MONTHLY", ip_country="GB"),
    _T("txn_ev33b", "cus_ev33", 220.00, 1 * _D, "SENTINEL*ADDON SEATS", ip_country="NG"),
    _T("txn_ev34a", "cus_ev34", 45.00, 40 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev34b", "cus_ev34", 45.00, 10 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev35a", "cus_ev35", 25.00, 30 * _D, "SENTINEL*PRO LITE"),
    _T("txn_ev35b", "cus_ev35", 60.00, 5 * _D, "SENTINEL*API CREDITS", card_last4="5555", ip_country="BR"),
    _T("txn_ev35c", "cus_ev35", 85.00, 4 * _D, "SENTINEL*HARDWARE KIT", card_last4="5555", ip_country="BR"),
    _T("txn_ev36a", "cus_ev36", 25.00, 30 * _D, "SENTINEL*PRO LITE", ip_country="NO"),
    _T("txn_ev36b", "cus_ev36", 60.00, 5 * _D, "SENTINEL*API CREDITS", card_last4="5555", ip_country="BR"),
    _T("txn_ev36c", "cus_ev36", 85.00, 4 * _D, "SENTINEL*HARDWARE KIT", card_last4="5555", ip_country="BR"),
    _T("txn_ev37a", "cus_ev37", 300.00, 40 * _D, "SENTINEL*ENTERPRISE MONTHLY"),
    _T("txn_ev37b", "cus_ev37", 1800.00, 1 * _D, "SENTINEL*HARDWARE BUNDLE", card_last4="6011", ip_country="CN"),
    _T("txn_ev39a", "cus_ev39", 49.00, 25 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev39b", "cus_ev39", 129.00, 3 * _D, "SENTINEL*API CREDITS", card_last4="3782", ip_country="RO"),
    # abnormal amount on the usual card/country (POL-FRD-001 "abnormal amounts")
    _T("txn_ev40a", "cus_ev40", 19.00, 70 * _D, "SENTINEL*PRO LITE"),
    _T("txn_ev40b", "cus_ev40", 19.00, 40 * _D, "SENTINEL*PRO LITE"),
    _T("txn_ev40c", "cus_ev40", 19.00, 10 * _D, "SENTINEL*PRO LITE"),
    _T("txn_ev40d", "cus_ev40", 2400.00, 2 * _D, "SENTINEL*HARDWARE BUNDLE"),
    _T("txn_ev41a", "cus_ev41", 129.00, 1 * _D, "SENTINEL*PRO ANNUAL", ip_country="CN"),
    _T("txn_ev42a", "cus_ev42", 29.00, 20 * _D, "SENTINEL*PRO MONTHLY", ip_country="DE"),
    _T("txn_ev42b", "cus_ev42", 12.00, 1 * _D, "SENTINEL*API CREDITS", card_last4="9134", ip_country="DE"),
    _T("txn_ev43a", "cus_ev43", 89.00, 1 * _D, "SENTINEL*HARDWARE KIT", ip_country="VN"),
    # --- refund window / eligibility -----------------------------------------
    _T("txn_ev51a", "cus_ev51", 19.00, 13 * _D, "SENTINEL*STARTER"),
    _T("txn_ev52a", "cus_ev52", 19.00, 15 * _D, "SENTINEL*STARTER"),
    _T("txn_ev53a", "cus_ev53", 49.00, 29 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev54a", "cus_ev54", 49.00, 31 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev55a", "cus_ev55", 180.00, 59 * _D, "SENTINEL*ENTERPRISE ADDON"),
    _T("txn_ev56a", "cus_ev56", 180.00, 61 * _D, "SENTINEL*ENTERPRISE ADDON"),
    _T("txn_ev57a", "cus_ev57", 2500.00, 40 * _D, "SENTINEL*ENTERPRISE PREMIUM"),
    _T("txn_ev58a", "cus_ev58", 200.00, 10 * _D, "SENTINEL*PRO ANNUAL LITE"),
    _T("txn_ev59a", "cus_ev59", 15.00, 50 * _D, "SENTINEL*STARTER"),
    _T("txn_ev59b", "cus_ev59", 15.00, 20 * _D, "SENTINEL*STARTER"),
    _T("txn_ev60a", "cus_ev60", 110.00, 150 * _D, "SENTINEL*PRO ONBOARDING"),
    _T("txn_ev61a", "cus_ev61", 90.00, 30 * _D, "SENTINEL*ENTERPRISE TRAINING", ip_country="PT"),
    # --- already refunded ----------------------------------------------------
    _T("txn_ev71a", "cus_ev71", 59.00, 12 * _D, "SENTINEL*PRO MONTHLY", status="REFUNDED", refunded_amount=59.00),
    _T("txn_ev72a", "cus_ev72", 35.00, 6 * _D, "SENTINEL*PRO SEATS"),
    _T("txn_ev72b", "cus_ev72", 35.00, 6 * _D - 1 * _M, "SENTINEL*PRO SEATS", status="REFUNDED", refunded_amount=35.00),
    _T("txn_ev73a", "cus_ev73", 250.00, 25 * _D, "SENTINEL*ENTERPRISE SEATS", status="REFUNDED", refunded_amount=250.00),
    _T("txn_ev74a", "cus_ev74", 780.00, 30 * _D, "SENTINEL*ENTERPRISE WORKSHOP", status="REFUNDED", refunded_amount=780.00),
    _T("txn_ev75a", "cus_ev75", 100.00, 10 * _D, "SENTINEL*PRO TEAM", status="PARTIALLY_REFUNDED", refunded_amount=40.00),
    _T("txn_ev76a", "cus_ev76", 85.00, 8 * _D, "SENTINEL*HARDWARE KIT", status="REFUNDED", ip_country="RU", refunded_amount=85.00),
    _T("txn_ev77a", "cus_ev77", 70.00, 100 * _D, "SENTINEL*PRO PLUS", status="REFUNDED", refunded_amount=70.00),
    _T("txn_ev78a", "cus_ev78", 25.00, 5 * _D, "SENTINEL*ADDON SEATS", status="REFUNDED", refunded_amount=25.00),
    _T("txn_ev78b", "cus_ev78", 40.00, 4 * _D, "SENTINEL*ADDON STORAGE"),
    # --- unknown customer / non-existent transaction ---------------------------
    _T("txn_ev81a", "cus_ev81", 40.00, 5 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev82a", "cus_ev82", 20.00, 8 * _D, "SENTINEL*ADDON SEATS"),
    _T("txn_ev82b", "cus_ev82", 40.00, 4 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev83a", "cus_ev83", 55.00, 6 * _D, "SENTINEL*PRO MONTHLY"),
    # --- prompt injection ------------------------------------------------------
    _T("txn_ev91a", "cus_ev91", 29.00, 3 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev91b", "cus_ev91", 29.00, 3 * _D - 1 * _M, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev92a", "cus_ev92", 39.00, 6 * _D, "SENTINEL*PRO LITE", ip_country="AR"),
    _T("txn_ev93a", "cus_ev93", 59.00, 5 * _D, "SENTINEL*PRO MONTHLY", ip_country="FR"),
    _T("txn_ev93b", "cus_ev93", 59.00, 5 * _D - 1 * _M, "SENTINEL*PRO MONTHLY", ip_country="FR"),
    _T("txn_ev94a", "cus_ev94", 60.00, 7 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev95a", "cus_ev95", 45.00, 4 * _D, "SENTINEL*PRO SEATS"),
    _T("txn_ev96a", "cus_ev96", 49.00, 3 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev97a", "cus_ev97", 88.00, 2 * _D, "SENTINEL*PRO PLUS"),
    _T("txn_ev97b", "cus_ev97", 88.00, 2 * _D - 1 * _M, "SENTINEL*PRO PLUS"),
    _T("txn_ev98a", "cus_ev98", 350.00, 10 * _D, "SENTINEL*ENTERPRISE SEATS"),
    _T("txn_ev99a", "cus_ev99", 33.00, 5 * _D, "SENTINEL*PRO LITE"),
    _T("txn_ev100a", "cus_ev100", 99.00, 60 * _D, "SENTINEL*STARTER YEARLY"),
    _T("txn_ev101a", "cus_ev101", 64.00, 4 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ev101b", "cus_ev101", 64.00, 4 * _D - 2 * _M, "SENTINEL*PRO MONTHLY"),
    # --- Spanish / mixed language ---------------------------------------------
    _T("txn_es01a", "cus_es01", 39.00, 3 * _D, "SENTINEL*PRO LITE", ip_country="ES"),
    _T("txn_es01b", "cus_es01", 39.00, 3 * _D - 1 * _M, "SENTINEL*PRO LITE", ip_country="ES"),
    _T("txn_es02a", "cus_es02", 25.00, 2 * _D, "SENTINEL*PRO MONTHLY", ip_country="AR"),
    _T("txn_es02b", "cus_es02", 25.00, 2 * _D - 2 * _M, "SENTINEL*PRO MONTHLY", ip_country="AR"),
    _T("txn_es03a", "cus_es03", 49.00, 30 * _D, "SENTINEL*PRO MONTHLY", ip_country="MX"),
    _T("txn_es03b", "cus_es03", 310.00, 2 * _D, "SENTINEL*HARDWARE KIT", card_last4="7788", ip_country="US"),
    _T("txn_es04a", "cus_es04", 99.00, 60 * _D, "SENTINEL*STARTER YEARLY", ip_country="CL"),
    _T("txn_es05a", "cus_es05", 59.00, 15 * _D, "SENTINEL*PRO MONTHLY", status="REFUNDED", ip_country="CO", refunded_amount=59.00),
    _T("txn_es06a", "cus_es06", 25.00, 1 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_es06b", "cus_es06", 25.00, 1 * _D - 1 * _M, "SENTINEL*PRO MONTHLY"),
    _T("txn_es07a", "cus_es07", 45.00, 40 * _D, "SENTINEL*PRO SEATS", ip_country="UY"),
    _T("txn_es07b", "cus_es07", 45.00, 10 * _D, "SENTINEL*PRO SEATS", ip_country="UY"),
    _T("txn_es08a", "cus_es08", 19.00, 3 * _D, "SENTINEL*STARTER", gateway_status="TIMEOUT", ip_country="AR"),
    _T("txn_es08b", "cus_es08", 19.00, 3 * _D - 20 * _M, "SENTINEL*STARTER", ip_country="AR"),
    _T("txn_es09a", "cus_es09", 19.00, 6 * _D, "SENTINEL*PRO LITE", ip_country="AR"),
    # --- ambiguous / incomplete --------------------------------------------------
    _T("txn_am01a", "cus_am01", 49.00, 5 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_am02a", "cus_am02", 49.00, 5 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_am03a", "cus_am03", 39.00, 4 * _D, "SENTINEL*ADDON SEATS"),
    _T("txn_am03b", "cus_am03", 89.00, 2 * _D, "SENTINEL*PRO PLUS"),
    _T("txn_am04a", "cus_am04", 49.00, 5 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_am05a", "cus_am05", 49.00, 5 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_am06a", "cus_am06", 15.00, 6 * _D, "SENTINEL*ADDON SEATS"),
    _T("txn_am07a", "cus_am07", 49.00, 3 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_am08a", "cus_am08", 129.00, 5 * _D, "SENTINEL*PRO ANNUAL LITE"),
    _T("txn_am09a", "cus_am09", 49.00, 5 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_am10a", "cus_am10", 59.00, 4 * _D, "SENTINEL*PRO MONTHLY"),
    # --- chaos ----------------------------------------------------------------------
    _T("txn_ch01a", "cus_ch01", 260.00, 3 * _D, "SENTINEL*ENTERPRISE ADDON"),
    _T("txn_ch01b", "cus_ch01", 260.00, 3 * _D - 1 * _M, "SENTINEL*ENTERPRISE ADDON"),
    _T("txn_ch03a", "cus_ch03", 75.00, 2 * _D, "SENTINEL*PRO MONTHLY"),
    _T("txn_ch03b", "cus_ch03", 75.00, 2 * _D - 1 * _M, "SENTINEL*PRO MONTHLY"),
)


async def seed_database(
    db: Database, *, force: bool = False, now: datetime | None = None, include_eval_fixtures: bool = False
) -> bool:
    """Insert seed rows. Returns True if seeding happened.

    ``include_eval_fixtures`` adds the per-scenario ledgers used by the golden eval set on top of
    the demo customers (which are always seeded unchanged).
    """
    customers = CUSTOMERS + EVAL_CUSTOMERS if include_eval_fixtures else CUSTOMERS
    transactions = TRANSACTIONS + EVAL_TRANSACTIONS if include_eval_fixtures else TRANSACTIONS
    now = now or utcnow()
    async with db.connect() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM customers")
        row = await cur.fetchone()
        count = int(row[0]) if row else 0
        if count and not force:
            return False
        if force:
            for table in (
                "refunds", "card_freezes", "crm_notes", "email_outbox", "supervisor_alerts", "inbound_emails",
                "stripe_mirror", "stripe_charges",  # Stripe-side objects stay; only the local mapping resets
                "ticket_events", "tickets", "transactions", "customers",
            ):
                await conn.execute(f"DELETE FROM {table}")  # noqa: S608 - static table names
        await conn.executemany(
            "INSERT INTO customers (id, email, name, tier, country, account_created_at, chargeback_count, account_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'ACTIVE')",
            [(c.id, c.email, c.name, c.tier, c.country, iso(now - c.age_days * _D), c.chargebacks) for c in customers],
        )
        await conn.executemany(
            "INSERT INTO transactions (id, customer_id, amount, currency, status, gateway_status, created_at, "
            "merchant_descriptor, card_last4, ip_country, refunded_amount) VALUES (?, ?, ?, 'USD', ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    t.id,
                    t.customer_id,
                    t.amount,
                    t.status,
                    t.gateway_status,
                    iso(now - t.ago),
                    t.descriptor,
                    t.card_last4,
                    t.ip_country,
                    t.refunded_amount,
                )
                for t in transactions
            ],
        )
        await conn.commit()
    return True
