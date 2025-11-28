"""Convert Netscape cookies.txt to twitter_cookies.json format"""

import json
import sys

def netscape_to_json(netscape_file, json_file):
    """Convert Netscape cookies.txt to JSON format"""
    cookies = {}
    
    with open(netscape_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse Netscape format: domain, flag, path, secure, expiration, name, value
            parts = line.split('\t')
            if len(parts) >= 7:
                name = parts[5]
                value = parts[6]
                cookies[name] = value
    
    # Save as JSON
    with open(json_file, 'w') as f:
        json.dump(cookies, f, indent=2)
    
    print(f"✓ Converted {len(cookies)} cookies")
    print(f"✓ Saved to: {json_file}")
    
    # Show important cookies
    important = ['auth_token', 'ct0', 'guest_id']
    found = [k for k in important if k in cookies]
    print(f"\n✓ Found important cookies: {', '.join(found)}")
    
    if 'auth_token' not in cookies:
        print("\n⚠️  WARNING: 'auth_token' cookie not found!")
        print("   Make sure you're logged into Twitter when exporting cookies")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_cookies.py cookies.txt [output.json]")
        print("\n1. Install browser extension: 'Get cookies.txt LOCALLY'")
        print("2. Go to twitter.com (make sure you're logged in)")
        print("3. Click extension and save as cookies.txt")
        print("4. Run: python convert_cookies.py cookies.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'twitter_cookies.json'
    
    netscape_to_json(input_file, output_file)
