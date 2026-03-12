import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), 'cartoon_products.db')

class DatabaseManager:
    def __init__(self):
        self.db_path = DB_PATH
        self.create_tables()
    
    def create_tables(self):
        """Create all necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Products table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_name TEXT NOT NULL,
                brand TEXT,
                category TEXT,
                keywords TEXT,
                description TEXT,
                expected_barcode TEXT,
                created_date DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Scan sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stage TEXT NOT NULL CHECK(stage IN ('inbound', 'sorting')),
                operator_notes TEXT,
                started_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                ended_at DATETIME
            )
        ''')

        # Scans table
        # quantity is nullable — NULL means flagged (inbound, qty unreadable)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ocr_text TEXT NOT NULL,
                barcode TEXT,
                matched_product_id INTEGER,
                match_confidence TEXT,
                match_score REAL,
                verified INTEGER DEFAULT 0,
                scan_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                device_id TEXT,
                notes TEXT,
                session_id INTEGER REFERENCES scan_sessions(id),
                quantity INTEGER,
                quantity_source TEXT DEFAULT 'default',
                FOREIGN KEY (matched_product_id) REFERENCES products(id)
            )
        ''')
        
        # Inventory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                quantity INTEGER DEFAULT 0,
                location TEXT,
                last_scanned DATETIME,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        ''')
        
        conn.commit()
        conn.close()

    # ==================== SESSION OPERATIONS ====================

    def create_session(self, stage, operator_notes=None):
        """Create a new scan session. Returns session ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO scan_sessions (stage, operator_notes)
            VALUES (?, ?)
        ''', (stage, operator_notes))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"✓ Session created: ID={session_id} stage={stage}")
        return session_id

    def close_session(self, session_id):
        """Mark a session as ended."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE scan_sessions SET ended_at = ? WHERE id = ?
        ''', (datetime.now(), session_id))
        conn.commit()
        conn.close()
        print(f"✓ Session {session_id} closed")

    def get_session(self, session_id):
        """Get a session by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM scan_sessions WHERE id = ?', (session_id,))
        session = cursor.fetchone()
        conn.close()
        return session

    # ==================== PRODUCT OPERATIONS ====================
    
    def add_product(self, product_name, brand, category, keywords, description, barcode=None):
        """Add a new product to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO products (product_name, brand, category, keywords, description, expected_barcode)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (product_name, brand, category, keywords, description, barcode))
        product_id = cursor.lastrowid
        cursor.execute('''
            INSERT INTO inventory (product_id, quantity, location)
            VALUES (?, ?, ?)
        ''', (product_id, 0, 'Main Warehouse'))
        conn.commit()
        conn.close()
        print(f"✓ Product added: {product_name} (ID: {product_id})")
        return product_id
    
    def get_all_products(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM products ORDER BY product_name')
        products = cursor.fetchall()
        conn.close()
        return products
    
    def get_product_by_id(self, product_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
        product = cursor.fetchone()
        conn.close()
        return product
    
    def view_all_products(self):
        products = self.get_all_products()
        if not products:
            print("\n⚠ No products in database!")
            return
        print("\n" + "="*70)
        print("PRODUCTS DATABASE")
        print("="*70)
        for p in products:
            print(f"\n[{p[0]}] {p[1]}")
            print(f"    Brand: {p[2]}")
            print(f"    Category: {p[3]}")
            print(f"    Keywords: {p[4]}")
            if p[6]:
                print(f"    Barcode: {p[6]}")
        print("="*70)
    
    # ==================== SCAN OPERATIONS ====================
    
    def save_scan(self, ocr_text, matched_product_id, match_confidence, match_score,
                  barcode=None, device_id=None, notes=None,
                  session_id=None, quantity=1, quantity_source='default'):
        """
        Save scan result to database.

        quantity        — units detected. NULL if flagged (inbound, qty unreadable).
        quantity_source — 'ocr'      : extracted from box text
                          'default'  : sorting fallback (1 per item)
                          'flagged'  : inbound, qty unknown — needs manual resolution
        session_id      — links this scan to an inbound/sorting session

        Inventory is only updated when:
        - scan is verified AND
        - quantity_source is not 'flagged'
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        verified = 1 if match_confidence in ['HIGH', 'MEDIUM'] else 0

        # Store NULL in DB when flagged so it's excluded from SUM() queries
        db_quantity = None if quantity_source == 'flagged' else quantity
        
        cursor.execute('''
            INSERT INTO scans (
                ocr_text, barcode, matched_product_id, match_confidence,
                match_score, verified, device_id, notes,
                session_id, quantity, quantity_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ocr_text, barcode, matched_product_id, match_confidence,
            match_score, verified, device_id, notes,
            session_id, db_quantity, quantity_source
        ))
        
        scan_id = cursor.lastrowid
        
        # Only update inventory if verified AND quantity is known
        if verified and matched_product_id and quantity_source != 'flagged':
            cursor.execute('''
                UPDATE inventory 
                SET quantity = quantity + ?,
                    last_scanned = ?
                WHERE product_id = ?
            ''', (quantity, datetime.now(), matched_product_id))
        
        conn.commit()
        conn.close()
        return scan_id

    def resolve_flagged_scan(self, scan_id, quantity):
        """
        Operator manually resolves a flagged scan by providing the quantity.
        Updates the scan record and adds quantity to inventory.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT matched_product_id, verified FROM scans WHERE id = ?
        ''', (scan_id,))
        row = cursor.fetchone()

        if not row:
            print(f"✗ Scan ID {scan_id} not found")
            conn.close()
            return False

        matched_product_id, verified = row

        cursor.execute('''
            UPDATE scans
            SET quantity = ?,
                quantity_source = 'resolved',
                notes = COALESCE(notes || ' | ', '') || 'qty resolved manually'
            WHERE id = ?
        ''', (quantity, scan_id))

        if verified and matched_product_id:
            cursor.execute('''
                UPDATE inventory
                SET quantity = quantity + ?,
                    last_scanned = ?
                WHERE product_id = ?
            ''', (quantity, datetime.now(), matched_product_id))
            print(f"✓ Inventory updated: +{quantity} units for product ID {matched_product_id}")

        conn.commit()
        conn.close()
        print(f"✓ Scan {scan_id} resolved with quantity {quantity}")
        return True

    # ==================== FLAGGED SCAN VIEWS ====================

    def get_flagged_scans(self):
        """Get all inbound scans where quantity could not be read."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.id, s.ocr_text, s.barcode, s.scan_date,
                   p.product_name, p.brand, ss.stage
            FROM scans s
            LEFT JOIN products p ON s.matched_product_id = p.id
            LEFT JOIN scan_sessions ss ON s.session_id = ss.id
            WHERE s.quantity_source = 'flagged'
            ORDER BY s.scan_date DESC
        ''')
        scans = cursor.fetchall()
        conn.close()
        return scans

    def view_flagged_scans(self):
        """Display all flagged scans that need manual quantity resolution."""
        scans = self.get_flagged_scans()

        print("\n" + "="*70)
        print("⚠ FLAGGED SCANS — QUANTITY UNKNOWN (Needs Manual Resolution)")
        print("="*70)

        if not scans:
            print("  ✓ No flagged scans — all quantities resolved!")
        else:
            for s in scans:
                print(f"\n[Scan ID: {s[0]}] {s[6].upper() if s[6] else 'UNKNOWN'} — {s[3]}")
                print(f"    Product:  {s[5]} {s[4]}" if s[4] else "    Product:  Unknown")
                print(f"    OCR Text: {s[1][:60]}")
                if s[2]:
                    print(f"    Barcode:  {s[2]}")
                print(f"    Action:   resolve_flagged_scan({s[0]}, <quantity>)")
            print(f"\n  Total flagged: {len(scans)} scan(s)")

        print("="*70)

    # ==================== QUANTITY SUMMARY ====================

    def get_quantity_by_stage(self, session_id=None):
        """
        Return total known quantity per product per stage.
        NULL (flagged) quantities are excluded from totals.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if session_id:
            cursor.execute('''
                SELECT ss.stage, p.product_name, p.brand, SUM(s.quantity) as total
                FROM scans s
                JOIN scan_sessions ss ON s.session_id = ss.id
                JOIN products p ON s.matched_product_id = p.id
                WHERE s.verified = 1
                  AND s.quantity IS NOT NULL
                  AND s.session_id = ?
                GROUP BY ss.stage, p.id
                ORDER BY ss.stage, total DESC
            ''', (session_id,))
        else:
            cursor.execute('''
                SELECT ss.stage, p.product_name, p.brand, SUM(s.quantity) as total
                FROM scans s
                JOIN scan_sessions ss ON s.session_id = ss.id
                JOIN products p ON s.matched_product_id = p.id
                WHERE s.verified = 1
                  AND s.quantity IS NOT NULL
                GROUP BY ss.stage, p.id
                ORDER BY ss.stage, total DESC
            ''')

        rows = cursor.fetchall()
        conn.close()
        return rows

    def view_quantity_summary(self, session_id=None):
        """Display quantity summary by stage, with flagged count warning."""
        rows = self.get_quantity_by_stage(session_id)
        flagged = self.get_flagged_scans()

        print("\n" + "="*70)
        if session_id:
            print(f"QUANTITY SUMMARY — Session {session_id}")
        else:
            print("QUANTITY SUMMARY — All Sessions")
        print("="*70)

        if not rows:
            print("  No verified scans with known quantity data yet.")
        else:
            current_stage = None
            stage_total = 0
            for stage, product_name, brand, total in rows:
                if stage != current_stage:
                    if current_stage is not None:
                        print(f"  {'─'*40}")
                        print(f"  Stage total (known): {stage_total} units")
                    print(f"\n[{stage.upper()}]")
                    current_stage = stage
                    stage_total = 0
                print(f"  {brand} {product_name}: {total} units")
                stage_total += total
            print(f"  {'─'*40}")
            print(f"  Stage total (known): {stage_total} units")

        if flagged:
            print(f"\n  ⚠ WARNING: {len(flagged)} flagged scan(s) with unknown quantity")
            print(f"  These are NOT included in totals above.")
            print(f"  Run view_flagged_scans() to see details.")

        print("="*70)

    # ==================== EXISTING SCAN VIEWS ====================
    
    def get_verified_scans(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.id, s.ocr_text, s.barcode, s.match_confidence, s.scan_date,
                   p.product_name, p.brand
            FROM scans s
            LEFT JOIN products p ON s.matched_product_id = p.id
            WHERE s.verified = 1
            ORDER BY s.scan_date DESC
        ''')
        scans = cursor.fetchall()
        conn.close()
        return scans
    
    def get_unverified_scans(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.id, s.ocr_text, s.barcode, s.match_confidence, s.scan_date,
                   s.notes
            FROM scans s
            WHERE s.verified = 0
            ORDER BY s.scan_date DESC
        ''')
        scans = cursor.fetchall()
        conn.close()
        return scans
    
    def get_scan_history(self, limit=10):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.id, s.ocr_text, s.verified, s.match_confidence, s.scan_date,
                   p.product_name, p.brand
            FROM scans s
            LEFT JOIN products p ON s.matched_product_id = p.id
            ORDER BY s.scan_date DESC
            LIMIT ?
        ''', (limit,))
        scans = cursor.fetchall()
        conn.close()
        return scans
    
    def view_verified_scans(self):
        scans = self.get_verified_scans()
        if not scans:
            print("\n⚠ No verified scans!")
            return
        print("\n" + "="*70)
        print(f"VERIFIED SCANS (✓ Status = 1)")
        print("="*70)
        for s in scans:
            print(f"\n[{s[0]}] {s[4]}")
            print(f"    Text: {s[1][:50]}...")
            print(f"    Confidence: {s[3]}")
            print(f"    Matched: {s[6]} - {s[5]}")
            if s[2]:
                print(f"    Barcode: {s[2]}")
        print("="*70)
    
    def view_unverified_scans(self):
        scans = self.get_unverified_scans()
        if not scans:
            print("\n✓ No unverified scans - all items matched!")
            return
        print("\n" + "="*70)
        print(f"UNVERIFIED SCANS (✗ Status = 0) - NEEDS ATTENTION")
        print("="*70)
        for s in scans:
            print(f"\n[{s[0]}] {s[4]}")
            print(f"    Text: {s[1][:50]}...")
            print(f"    Confidence: {s[3]}")
            if s[2]:
                print(f"    Barcode: {s[2]}")
            if s[5]:
                print(f"    Notes: {s[5]}")
        print("="*70)
        print(f"\nTotal unverified: {len(scans)} items")
    
    def view_scan_history(self, limit=10):
        scans = self.get_scan_history(limit)
        if not scans:
            print("\n⚠ No scan history!")
            return
        print("\n" + "="*70)
        print(f"SCAN HISTORY (Last {limit})")
        print("="*70)
        for s in scans:
            status_icon = "✓" if s[2] == 1 else "✗"
            print(f"\n[{s[0]}] {status_icon} {s[4]}")
            print(f"    Text: {s[1][:50]}...")
            print(f"    Status: {'Verified' if s[2] == 1 else 'Unverified'}")
            print(f"    Confidence: {s[3]}")
            if s[5]:
                print(f"    Matched: {s[6]} - {s[5]}")
        print("="*70)
    
    # ==================== INVENTORY OPERATIONS ====================
    
    def get_inventory(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT p.id, p.product_name, p.brand, i.quantity, i.location, i.last_scanned
            FROM inventory i
            JOIN products p ON i.product_id = p.id
            ORDER BY i.quantity DESC
        ''')
        inventory = cursor.fetchall()
        conn.close()
        return inventory
    
    def view_inventory(self):
        inventory = self.get_inventory()
        print("\n" + "="*70)
        print("INVENTORY REPORT")
        print("="*70)
        for item in inventory:
            print(f"\n[{item[0]}] {item[1]} ({item[2]})")
            print(f"    Quantity: {item[3]}")
            print(f"    Location: {item[4]}")
            print(f"    Last Scanned: {item[5] if item[5] else 'Never'}")
        print("="*70)
    
    # ==================== STATISTICS ====================
    
    def get_statistics(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        stats = {}
        cursor.execute('SELECT COUNT(*) FROM products')
        stats['total_products'] = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM scans')
        stats['total_scans'] = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM scans WHERE verified = 1')
        stats['verified_count'] = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM scans WHERE verified = 0')
        stats['unverified_count'] = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM scans WHERE quantity_source = ?', ('flagged',))
        stats['flagged_count'] = cursor.fetchone()[0]
        cursor.execute('''
            SELECT match_confidence, COUNT(*) 
            FROM scans 
            GROUP BY match_confidence
        ''')
        stats['by_confidence'] = cursor.fetchall()
        cursor.execute('SELECT SUM(quantity) FROM inventory')
        stats['total_inventory'] = cursor.fetchone()[0] or 0
        cursor.execute('''
            SELECT ss.stage, SUM(s.quantity)
            FROM scans s
            JOIN scan_sessions ss ON s.session_id = ss.id
            WHERE s.verified = 1
              AND s.quantity IS NOT NULL
            GROUP BY ss.stage
        ''')
        stats['quantity_by_stage'] = dict(cursor.fetchall())
        if stats['total_scans'] > 0:
            stats['verification_rate'] = (stats['verified_count'] / stats['total_scans']) * 100
        else:
            stats['verification_rate'] = 0
        conn.close()
        return stats
    
    def view_statistics(self):
        stats = self.get_statistics()
        print("\n" + "="*70)
        print("DATABASE STATISTICS")
        print("="*70)
        print(f"Total Products: {stats['total_products']}")
        print(f"Total Scans: {stats['total_scans']}")
        print(f"Total Inventory: {stats['total_inventory']} items")
        print(f"\nVerification Status:")
        print(f"  ✓ Verified (1): {stats['verified_count']}")
        print(f"  ✗ Unverified (0): {stats['unverified_count']}")
        print(f"  ⚠ Flagged (qty unknown): {stats['flagged_count']}")
        print(f"  Verification Rate: {stats['verification_rate']:.1f}%")
        print(f"\nScans by Confidence:")
        for conf in stats['by_confidence']:
            print(f"  {conf[0]}: {conf[1]}")
        print(f"\nQuantity by Stage (known only):")
        if stats.get('quantity_by_stage'):
            for stage, qty in stats['quantity_by_stage'].items():
                print(f"  {stage}: {qty} units")
        else:
            print(f"  No quantity data yet")
        print("="*70)
    
    # ==================== UTILITY ====================
    
    def get_unverified_list(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM scans WHERE verified = 0')
        unverified_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return unverified_ids
    
    def get_verified_list(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM scans WHERE verified = 1')
        verified_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return verified_ids

    # ==================== INITIALIZATION ====================
    
    def add_sample_products(self):
        sample_products = [
            ('Sephora Lipstick V2', 'Sephora', 'Cosmetics',
             'lipstick, v2, makeup, 5ephora, sphora',
             'Sephora brand lipstick version 2', '1234567890'),
            ('L\'Oreal Paris Rouge Signature', 'L\'Oreal', 'Cosmetics',
             'loreal, rouge, lipstick, matte, signature',
             'Matte liquid lipstick', '9876543210'),
            ('Maybelline SuperStay', 'Maybelline', 'Cosmetics',
             'maybelline, superstay, lipstick, long-lasting, maybeline',
             '24-hour wear lipstick', '5555555555'),
        ]
        for p in sample_products:
            self.add_product(p[0], p[1], p[2], p[3], p[4], p[5])
        print("\n✓ Sample products added successfully!")

    def delete_scan(self, scan_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM scans WHERE id = ?', (scan_id,))
        conn.commit()
        conn.close()

    def update_scan(self, scan_id, match_confidence=None, verified=None, notes=None):
        fields = []
        values = []
        if match_confidence is not None:
            fields.append('match_confidence = ?')
            values.append(match_confidence)
        if verified is not None:
            fields.append('verified = ?')
            values.append(verified)
        if notes is not None:
            fields.append('notes = ?')
            values.append(notes)
        if not fields:
            return
        values.append(scan_id)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            f'UPDATE scans SET {", ".join(fields)} WHERE id = ?',
            values
        )
        conn.commit()
        conn.close()


if __name__ == "__main__":
    print("\n### DATABASE MANAGER TEST ###")
    db = DatabaseManager()

    print("\n1. Creating inbound session...")
    session_id = db.create_session(stage='inbound')

    print("\n2. Scan WITH quantity (OCR read it)...")
    db.save_scan(
        ocr_text="SEPHORA Cream Lip Stain Qty: 4",
        matched_product_id=1,
        match_confidence="HIGH",
        match_score=100.0,
        session_id=session_id,
        quantity=4,
        quantity_source='ocr'
    )

    print("\n3. Scan WITHOUT quantity (flagged)...")
    db.save_scan(
        ocr_text="SEPHORA Cream Lip Gloss",
        matched_product_id=2,
        match_confidence="HIGH",
        match_score=100.0,
        session_id=session_id,
        quantity=None,
        quantity_source='flagged'
    )

    print("\n4. Quantity summary (flagged excluded)...")
    db.view_quantity_summary(session_id=session_id)

    print("\n5. View flagged scans...")
    db.view_flagged_scans()

    print("\n6. Resolve flagged scan manually...")
    db.resolve_flagged_scan(scan_id=2, quantity=6)

    print("\n7. Quantity summary after resolution...")
    db.view_quantity_summary(session_id=session_id)

    print("\n8. Statistics...")
    db.view_statistics()