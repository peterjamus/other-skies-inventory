# Other Skies Rare Books - Inventory Management System

## Local-First Professional Book Processing Platform

A powerful, locally-deployable system for processing rare book inventory with archival image standards, intelligent metadata extraction, multi-marketplace export, and integrated shipping management.

### Core Features
- **Archival Image Processing**: LUT-based color grading only, maintaining original integrity
- **Intelligent OCR & Metadata**: Multi-source API enrichment with manual override
- **Pro-Level Tools**: Keyboard shortcuts, batch operations, custom workflows
- **Multi-Marketplace Export**: AbeBooks, Biblio, Shopify, eBay formatted CSVs
- **Shipping Integration**: Label generation, insurance automation, tracking management
- **Local-First Architecture**: Runs entirely on your machine, no cloud dependency

### System Requirements
- Node.js 20+ 
- Python 3.10+ (for OCR processing)
- PostgreSQL 15+ (local instance)
- 8GB RAM minimum
- 50GB storage for image archives

### Quick Start
```bash
# Clone and setup
git clone https://github.com/peterjamus/other-skies-inventory.git
cd other-skies-inventory
npm install
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start local services
npm run db:start      # Starts local PostgreSQL
npm run dev          # Starts Next.js frontend
python ocr_server.py # Starts OCR service
```

### Required API Keys (store in .env.local)
```
# Free Tier Services
OPENLIB_API_KEY=none_required
GOOGLE_BOOKS_API_KEY=your_key_here

# Paid Services (optional initially)
ISBNDB_API_KEY=your_key_here
SHIPPO_API_KEY=your_key_here
UPIC_API_KEY=your_key_here

# Local Database
DATABASE_URL=postgresql://localhost:5432/otherskies
```
