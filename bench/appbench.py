"""App probe: hand-written queries over SaaS-style tables (invoices, issues, CRM, orders, staff).

FilterBench tables are Wikipedia tables. Filter boxes live in apps, so this checks the same
model on tables shaped like the ones it would actually meet. Queries were written by hand
(by Claude, for scoring only) before looking at any model output; "or" marks or-lists.

  python3 bench/appbench.py            # writes bench/out/fb.app.json, score it like any FilterBench file
"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from filterbench import ROOT, col_numeric, run  # noqa: E402

R = random.Random(7)
pick = R.choice


def table(header, n, row):
    rows = [[str(v) for v in row(i)] for i in range(n)]
    return {"header": header, "rows": rows, "types": ["real" if col_numeric(rows, j) else "text" for j in range(len(header))]}


CLIENTS = ["Acme Corp", "Globex", "Initech", "Umbrella Labs", "Hooli", "Stark Industries", "Wayne Enterprises", "Soylent", "Vandelay Industries", "Pied Piper"]
PEOPLE = ["Maria Chen", "Tom Becker", "Aisha Khan", "Lucas Silva", "Priya Patel", "Jonas Weber", "Sofia Rossi", "Daniel Kim"]
CITIES = ["Berlin", "Austin", "Toronto", "Lisbon", "New York", "San Francisco", "London", "Singapore"]
TABLES = {
    "invoices": table(["Invoice", "Client", "Balance", "Days late", "Status", "Due date"], 40, lambda i: [
        f"INV-{1001 + i}", CLIENTS[i % 10], [120, 450, 980, 1500, 2400, 5200, 8800, 12500][(i * 3) % 8], [0, 0, 3, 12, 31, 45, 60, 95][(i * 5) % 8],
        ["paid", "overdue", "pending", "draft"][i % 4], f"2026-{1 + i % 12:02d}-{1 + (i * 7) % 28:02d}"]),
    "issues": table(["ID", "Title", "Assignee", "Priority", "Status", "Story points", "Sprint"], 40, lambda i: [
        f"ENG-{200 + i}", ["Login page crashes on Safari", "Add dark mode toggle", "Export to CSV is slow", "Fix typo in billing email", "Upgrade payment SDK",
                           "Search returns stale results", "Onboarding checklist redesign", "Webhook retries fail silently"][i % 8] + (f" ({i})" if i >= 8 else ""),
        PEOPLE[(i * 3) % 8], ["low", "medium", "high", "urgent"][(i * 3) % 4], ["open", "in progress", "blocked", "closed", "in review"][i % 5], [1, 2, 3, 5, 8, 13][i % 6], 20 + i // 8]),
    "customers": table(["Name", "Company", "Country", "Plan", "MRR", "Seats", "Owner"], 40, lambda i: [
        ["Olivia Brown", "Noah Müller", "Emma García", "Liam O'Connor", "Ava Nakamura", "Ethan Dubois", "Mia Kowalski", "Leo Andersson"][i % 8] + ("" if i < 8 else f" {'I' * (i // 8)}"),
        CLIENTS[(i * 7) % 10], ["Germany", "United States", "Canada", "Portugal", "Japan", "France", "Brazil"][(i * 3) % 7], ["free", "starter", "pro", "enterprise"][i % 4],
        [0, 29, 99, 499, 1200, 4800][(i * 5) % 6], [1, 3, 5, 12, 40, 250][(i * 7) % 6], PEOPLE[(i // 4) % 4]]),
    "orders": table(["Order", "Customer", "Product", "Qty", "Total", "Shipping", "State", "City"], 40, lambda i: [
        f"#{50210 + i}", PEOPLE[(i * 5) % 8], ["Wireless Mouse", "Mechanical Keyboard", "USB-C Hub", "Laptop Stand", "Noise Cancelling Headphones", "Webcam HD"][i % 6],
        [1, 1, 2, 3, 5, 10][(i * 5) % 6], [19.99, 49.5, 89, 129, 240, 799.99, 1450][(i * 3) % 7], ["standard", "express", "pickup"][i % 3],
        ["delivered", "shipped", "processing", "cancelled", "refunded"][(i * 3) % 5], CITIES[(i * 3) % 8]]),
    "staff": table(["Name", "Department", "Title", "Location", "Salary", "Start year", "Remote"], 40, lambda i: [
        PEOPLE[i % 8] if i < 8 else f"{pick(['Anna', 'Ben', 'Carla', 'Dev', 'Elena', 'Farid'])} {pick(['Novak', 'Okafor', 'Lindqvist', 'Tanaka', 'Moreau', 'Haddad'])} {i}",
        ["Engineering", "Sales", "Marketing", "Support", "Finance"][i % 5], ["Software Engineer", "Account Executive", "Designer", "Support Specialist", "Analyst", "Engineering Manager"][(i * 5) % 6],
        CITIES[(i * 5) % 8], [48000, 62000, 75000, 91000, 118000, 145000, 180000][(i * 3) % 7], 2012 + (i * 5) % 14, ["yes", "no"][i % 2]]),
}

# (table, query, [(column name, op, value), ...], tag)
Q = [
    ("invoices", "overdue", [("Status", "=", "overdue")], ""),
    ("invoices", "overdue over 5k", [("Status", "=", "overdue"), ("Balance", ">", "5000")], ""),
    ("invoices", "acme invoices that are still pending", [("Client", "=", "Acme Corp"), ("Status", "=", "pending")], ""),
    ("invoices", "balance >= 2400", [("Balance", ">=", "2400")], ""),
    ("invoices", "more than 30 days late", [("Days late", ">", "30")], ""),
    ("invoices", "not paid", [("Status", "!=", "paid")], ""),
    ("invoices", "client: hooli, status: pending", [("Client", "=", "Hooli"), ("Status", "=", "pending")], ""),
    ("invoices", "show me globex invoices under $1,000", [("Client", "=", "Globex"), ("Balance", "<", "1000")], ""),
    ("invoices", "late by at least 45 days and balance over 500", [("Days late", ">=", "45"), ("Balance", ">", "500")], ""),
    ("invoices", "industries", [("Client", "contains", "industries")], ""),
    ("invoices", "INV-1012", [("Invoice", "=", "INV-1012")], ""),
    ("invoices", "balance between 1000 and 5200", [("Balance", ">=", "1000"), ("Balance", "<=", "5200")], ""),
    ("invoices", "overdue or pending", [("Status", "=", "overdue"), ("Status", "=", "pending")], "or"),
    ("invoices", "stark or wayne, overdue", [("Client", "=", "Stark Industries"), ("Client", "=", "Wayne Enterprises"), ("Status", "=", "overdue")], "or"),
    ("invoices", "everything except drafts", [("Status", "!=", "draft")], ""),
    ("invoices", "0 days late", [("Days late", "=", "0")], ""),
    ("issues", "urgent", [("Priority", "=", "urgent")], ""),
    ("issues", "open issues assigned to maria chen", [("Status", "=", "open"), ("Assignee", "=", "Maria Chen")], ""),
    ("issues", "blocked high priority", [("Status", "=", "blocked"), ("Priority", "=", "high")], ""),
    ("issues", "assignee: tom becker", [("Assignee", "=", "Tom Becker")], ""),
    ("issues", "story points > 5", [("Story points", ">", "5")], ""),
    ("issues", "sprint 22 not closed", [("Sprint", "=", "22"), ("Status", "!=", "closed")], ""),
    ("issues", "title has safari", [("Title", "contains", "safari")], ""),
    ("issues", "anything about billing", [("Title", "contains", "billing")], ""),
    ("issues", "in progress", [("Status", "=", "in progress")], ""),
    ("issues", "priya's issues with 8 or more points", [("Assignee", "=", "Priya Patel"), ("Story points", ">=", "8")], ""),
    ("issues", "priority=low, status=in review", [("Priority", "=", "low"), ("Status", "=", "in review")], ""),
    ("issues", "high or urgent", [("Priority", "=", "high"), ("Priority", "=", "urgent")], "or"),
    ("issues", "open or blocked in sprint 21", [("Status", "=", "open"), ("Status", "=", "blocked"), ("Sprint", "=", "21")], "or"),
    ("issues", "ENG-215", [("ID", "=", "ENG-215")], ""),
    ("issues", "small tasks, 2 points or less", [("Story points", "<=", "2")], ""),
    ("issues", "not assigned to daniel kim", [("Assignee", "!=", "Daniel Kim")], ""),
    ("customers", "enterprise", [("Plan", "=", "enterprise")], ""),
    ("customers", "german customers on pro", [("Country", "=", "Germany"), ("Plan", "=", "pro")], ""),
    ("customers", "mrr over 1k", [("MRR", ">", "1000")], ""),
    ("customers", "customers in japan", [("Country", "=", "Japan")], ""),
    ("customers", "more than 40 seats", [("Seats", ">", "40")], ""),
    ("customers", "owner aisha khan, plan free", [("Owner", "=", "Aisha Khan"), ("Plan", "=", "free")], ""),
    ("customers", "company = pied piper", [("Company", "=", "Pied Piper")], ""),
    ("customers", "paying customers", [("MRR", ">", "0")], ""),
    ("customers", "not in the united states", [("Country", "!=", "United States")], ""),
    ("customers", "at least 12 seats and mrr under 500", [("Seats", ">=", "12"), ("MRR", "<", "500")], ""),
    ("customers", "canada or brazil", [("Country", "=", "Canada"), ("Country", "=", "Brazil")], "or"),
    ("customers", "pro or enterprise in france", [("Plan", "=", "pro"), ("Plan", "=", "enterprise"), ("Country", "=", "France")], "or"),
    ("customers", "muller", [("Name", "contains", "müller")], ""),
    ("customers", "starter plan with 3 seats", [("Plan", "=", "starter"), ("Seats", "=", "3")], ""),
    ("customers", "Seats<=3", [("Seats", "<=", "3")], ""),
    ("customers", "accounts owned by lucas", [("Owner", "=", "Lucas Silva")], ""),
    ("orders", "cancelled", [("State", "=", "cancelled")], ""),
    ("orders", "express orders to berlin", [("Shipping", "=", "express"), ("City", "=", "Berlin")], ""),
    ("orders", "total over 200", [("Total", ">", "200")], ""),
    ("orders", "keyboard", [("Product", "contains", "keyboard")], ""),
    ("orders", "qty at least 5", [("Qty", ">=", "5")], ""),
    ("orders", "orders from sofia rossi that were delivered", [("Customer", "=", "Sofia Rossi"), ("State", "=", "delivered")], ""),
    ("orders", "shipping: pickup, city: london", [("Shipping", "=", "pickup"), ("City", "=", "London")], ""),
    ("orders", "not delivered", [("State", "!=", "delivered")], ""),
    ("orders", "headphones under 1000", [("Product", "contains", "headphones"), ("Total", "<", "1000")], ""),
    ("orders", "refunded or cancelled", [("State", "=", "refunded"), ("State", "=", "cancelled")], "or"),
    ("orders", "new york or san francisco, express", [("City", "=", "New York"), ("City", "=", "San Francisco"), ("Shipping", "=", "express")], "or"),
    ("orders", "#50233", [("Order", "=", "#50233")], ""),
    ("orders", "usb-c hub", [("Product", "=", "USB-C Hub")], ""),
    ("orders", "big orders, 10 items", [("Qty", "=", "10")], ""),
    ("orders", "processing orders above $100 shipped standard", [("State", "=", "processing"), ("Total", ">", "100"), ("Shipping", "=", "standard")], ""),
    ("orders", "city is not toronto", [("City", "!=", "Toronto")], ""),
    ("staff", "engineering", [("Department", "=", "Engineering")], ""),
    ("staff", "remote engineers", [("Remote", "=", "yes"), ("Department", "=", "Engineering")], ""),
    ("staff", "salary over 100k", [("Salary", ">", "100000")], ""),
    ("staff", "people in lisbon", [("Location", "=", "Lisbon")], ""),
    ("staff", "started before 2018", [("Start year", "<", "2018")], ""),
    ("staff", "designers", [("Title", "=", "Designer")], ""),
    ("staff", "sales team in london", [("Department", "=", "Sales"), ("Location", "=", "London")], ""),
    ("staff", "department: finance, salary >= 75k", [("Department", "=", "Finance"), ("Salary", ">=", "75000")], ""),
    ("staff", "managers", [("Title", "contains", "manager")], ""),
    ("staff", "not remote", [("Remote", "=", "no")], ""),
    ("staff", "joined 2020 or later", [("Start year", ">=", "2020")], ""),
    ("staff", "support or marketing", [("Department", "=", "Support"), ("Department", "=", "Marketing")], "or"),
    ("staff", "berlin or austin, salary under 80k", [("Location", "=", "Berlin"), ("Location", "=", "Austin"), ("Salary", "<", "80000")], "or"),
    ("staff", "analysts earning at most 91000", [("Title", "=", "Analyst"), ("Salary", "<=", "91000")], ""),
    ("staff", "everyone outside engineering", [("Department", "!=", "Engineering")], ""),
    ("staff", "start year 2015", [("Start year", "=", "2015")], ""),
]

if __name__ == "__main__":
    examples = []
    for t, q, conds, tag in Q:
        tab = TABLES[t]
        cs = [[tab["header"].index(h), op, v] for h, op, v in conds]
        hit = run(tab["rows"], cs, [x == "real" for x in tab["types"]])
        assert 0 < len(hit) < len(tab["rows"]), (q, len(hit))  # every query must actually filter
        examples.append({"table": t, "conds": cs, "q": q, "tag": tag})
    json.dump({"model": "hand-written", "split": "probe", "examples": examples, "tables": TABLES}, open(ROOT / "fb.app.json", "w"))
    print(f"wrote {len(examples)} queries over {len(TABLES)} tables ({sum(e['tag'] == 'or' for e in examples)} with or-lists)")
