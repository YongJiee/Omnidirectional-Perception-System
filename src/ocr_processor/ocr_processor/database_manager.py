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
        
        # Scans table - simplified
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
        
        # Initialize inventory
        cursor.execute('''
            INSERT INTO inventory (product_id, quantity, location)
            VALUES (?, ?, ?)
        ''', (product_id, 0, 'Main Warehouse'))
        
        conn.commit()
        conn.close()
        
        print(f"✓ Product added: {product_name} (ID: {product_id})")
        return product_id
    
    def get_all_products(self):
        """Get all products"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM products ORDER BY product_name')
        products = cursor.fetchall()
        conn.close()
        return products
    
    def get_product_by_id(self, product_id):
        """Get specific product"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM products WHERE id = ?', (product_id,))
        product = cursor.fetchone()
        conn.close()
        return product
    
    def view_all_products(self):
        """Display all products"""
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
            if p[6]:  # barcode
                print(f"    Barcode: {p[6]}")
        print("="*70)
    
    # ==================== SCAN OPERATIONS ====================
    
    def save_scan(self, ocr_text, matched_product_id, match_confidence, match_score,
                  barcode=None, device_id=None, notes=None):
        """
        Save scan result to database
        
        verified status:
        - 1 if matched (HIGH or MEDIUM confidence)
        - 0 if not matched or LOW confidence
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Determine verified status
        verified = 1 if match_confidence in ['HIGH', 'MEDIUM'] else 0
        
        cursor.execute('''
            INSERT INTO scans (
                ocr_text, barcode, matched_product_id, match_confidence, 
                match_score, verified, device_id, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            ocr_text, barcode, matched_product_id, match_confidence,
            match_score, verified, device_id, notes
        ))
        
        scan_id = cursor.lastrowid
        
        # Update inventory only if verified
        if verified and matched_product_id:
            cursor.execute('''
                UPDATE inventory 
                SET quantity = quantity + 1, 
                    last_scanned = ?
                WHERE product_id = ?
            ''', (datetime.now(), matched_product_id))
        
        conn.commit()
        conn.close()
        
        return scan_id
    
    def get_verified_scans(self):
        """Get all verified scans (status = 1)"""
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
        """Get all unverified scans (status = 0) - needs attention"""
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
        """Get recent scans"""
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
        """Display verified scans"""
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
        """Display unverified scans - items that need attention"""
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
        """Display scan history"""
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
        """Get all inventory"""
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
        """Display inventory"""
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
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total products
        cursor.execute('SELECT COUNT(*) FROM products')
        stats['total_products'] = cursor.fetchone()[0]
        
        # Total scans
        cursor.execute('SELECT COUNT(*) FROM scans')
        stats['total_scans'] = cursor.fetchone()[0]
        
        # Verified vs unverified
        cursor.execute('SELECT COUNT(*) FROM scans WHERE verified = 1')
        stats['verified_count'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM scans WHERE verified = 0')
        stats['unverified_count'] = cursor.fetchone()[0]
        
        # By confidence
        cursor.execute('''
            SELECT match_confidence, COUNT(*) 
            FROM scans 
            GROUP BY match_confidence
        ''')
        stats['by_confidence'] = cursor.fetchall()
        
        # Total inventory
        cursor.execute('SELECT SUM(quantity) FROM inventory')
        stats['total_inventory'] = cursor.fetchone()[0] or 0
        
        # Verification rate
        if stats['total_scans'] > 0:
            stats['verification_rate'] = (stats['verified_count'] / stats['total_scans']) * 100
        else:
            stats['verification_rate'] = 0
        
        conn.close()
        return stats
    
    def view_statistics(self):
        """Display statistics"""
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
        print(f"  Verification Rate: {stats['verification_rate']:.1f}%")
        
        print(f"\nScans by Confidence:")
        for conf in stats['by_confidence']:
            print(f"  {conf[0]}: {conf[1]}")
        
        print("="*70)
    
    # ==================== UTILITY ====================
    
    def get_unverified_list(self):
        """
        Get simple list of unverified scan IDs
        Useful for your code to know which items need attention
        
        Returns: list of scan IDs with verified=0
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM scans WHERE verified = 0')
        unverified_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return unverified_ids
    
    def get_verified_list(self):
        """
        Get simple list of verified scan IDs
        
        Returns: list of scan IDs with verified=1
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM scans WHERE verified = 1')
        verified_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        return verified_ids
    
    # ==================== INITIALIZATION ====================
    
    def add_sample_products(self):
        """Add sample products for testing"""
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


# Test the database manager
if __name__ == "__main__":
    print("\n### DATABASE MANAGER TEST ###")
    
    db = DatabaseManager()
    
    print("\n1. Adding sample products...")
    db.add_sample_products()
    
    print("\n2. Viewing products...")
    db.view_all_products()
    
    print("\n3. Simulating some scans...")
    # Simulate a successful scan
    db.save_scan(
        ocr_text="SEPHORA Lipstick V2",
        matched_product_id=1,
        match_confidence="HIGH",
        match_score=180.5,
        barcode="1234567890",
        device_id="PI-001"
    )
    
    # Simulate a failed scan
    db.save_scan(
        ocr_text="Unknown Product XYZ",
        matched_product_id=None,
        match_confidence="NO MATCH",
        match_score=25.3,
        device_id="PI-001"
    )
    
    print("\n4. Viewing verified scans...")
    db.view_verified_scans()
    
    print("\n5. Viewing unverified scans...")
    db.view_unverified_scans()
    
    print("\n6. Getting unverified list for code...")
    unverified_ids = db.get_unverified_list()
    print(f"Unverified IDs: {unverified_ids}")
    
    print("\n7. Viewing statistics...")
    db.view_statistics()