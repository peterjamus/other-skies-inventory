# QUICKSTART GUIDE - Other Skies Inventory System

## Alpha Version - Local Deployment

This system runs entirely on your local machine for testing before cloud deployment.

## Prerequisites Installation

### 1. Install Required Software

#### macOS:
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install node@20 python@3.10 postgresql@15 tesseract

# Install Python pip
python3 -m ensurepip --upgrade
```

#### Ubuntu/Debian:
```bash
# Update package list
sudo apt update

# Install dependencies
sudo apt install -y nodejs npm python3 python3-pip postgresql tesseract-ocr

# Install Node 20 (if default is older)
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

#### Windows:
1. Install Node.js 20: https://nodejs.org/
2. Install Python 3.10+: https://www.python.org/downloads/
3. Install PostgreSQL 15: https://www.postgresql.org/download/windows/
4. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
5. Install Git Bash: https://git-scm.com/downloads

## Quick Setup (5 minutes)

### Step 1: Clone/Download the Code
```bash
# If using git
git clone https://github.com/peterjamus/other-skies-inventory.git
cd other-skies-inventory

# Or just extract the zip file and navigate to it
cd other-skies-inventory
```

### Step 2: Run Automatic Setup
```bash
./setup.sh
```

This will:
- Check all prerequisites
- Install dependencies
- Create database
- Set up directories
- Generate configuration

### Step 3: Get Your API Keys (Free)

#### Google Books API (Required - Free):
1. Go to https://console.cloud.google.com/
2. Create a new project (or select existing)
3. Enable "Books API"
4. Create credentials → API Key
5. Copy the key

#### Edit .env.local:
```bash
nano .env.local
# Or use any text editor
```

Add your Google Books API key:
```
GOOGLE_BOOKS_API_KEY="your-key-here"
```

### Step 4: Start the System

**Terminal 1 - Database:**
```bash
npm run db:start
```

**Terminal 2 - Web Interface:**
```bash
npm run dev
```

**Terminal 3 - OCR Service:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python ocr_server.py
```

**Terminal 4 - LUT Service:**
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate  
python lut_processor.py
```

### Step 5: Open Browser
Navigate to: http://localhost:3000

## First Test Run

### 1. Prepare Test Images
- Take 3-5 photos of book covers/title pages
- Save them to a folder on your desktop
- Images should be well-lit with clear text

### 2. Upload Images
- Click "Processing Center" in the nav
- Drag your images to the upload zone
- Or press Ctrl+U and select files

### 3. Process Batch
- Select your LUT preference (start with "neutral_accurate")
- Click "Start Processing"
- Wait for OCR and metadata extraction (30 seconds per book)

### 4. Review & Edit
- Use arrow keys to navigate between books
- AI suggestions appear in yellow
- Press Enter to approve and move to next
- Edit any incorrect fields

### 5. Export to Marketplaces
- Go to "Export" tab
- Select target marketplaces
- Download CSV files
- Upload to respective platforms

## LUT Options Explained

- **neutral_accurate** - No color shift, maximum fidelity (recommended for start)
- **warm_archival** - Subtle golden tone for vintage books
- **gothic_drama** - Higher contrast for atmospheric weird fiction
- **parchment** - Aged paper effect for antique books

## Pro Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| Enter | Approve current book & next |
| Ctrl+Enter | Approve all ready books |
| ← / → | Navigate between books |
| Ctrl+U | Quick upload |
| Ctrl+L | Focus LUT selector |
| Tab | Next field in form |
| Shift+Tab | Previous field |

## Troubleshooting

### "Cannot connect to database"
```bash
# Make sure PostgreSQL is running
npm run db:start

# If that fails, try:
pg_ctl -D ./data/postgres restart
```

### "OCR service not responding"
```bash
# Check if service is running
curl http://localhost:8001/health

# Restart OCR service
# Kill the Python process (Ctrl+C) then:
source venv/bin/activate
python ocr_server.py
```

### "Images not processing"
- Check image format (JPEG, PNG, TIFF supported)
- Ensure images are under 10MB each
- Verify Tesseract is installed: `tesseract --version`

### "LUT not applying"
```bash
# Check LUT service
curl http://localhost:8002/available-luts

# Restart if needed
```

## Database Management

### View your data:
```bash
npx prisma studio
# Opens at http://localhost:5555
```

### Backup database:
```bash
pg_dump otherskies > backup_$(date +%Y%m%d).sql
```

### Restore database:
```bash
psql otherskies < backup_20240101.sql
```

## Shipping Integration (Optional)

### To enable shipping labels:

1. Sign up for Shippo (free): https://goshippo.com/
2. Get API key from dashboard
3. Add to .env.local:
```
SHIPPO_API_KEY="shippo_test_..."
```

### To enable insurance:

1. Sign up for U-PIC: https://u-pic.com/
2. Get API credentials
3. Add to .env.local:
```
UPIC_API_KEY="your-key"
```

## Project Structure

```
other-skies-inventory/
├── app/                    # Next.js app pages
│   ├── processing/        # Batch processing interface
│   ├── inventory/         # Book management
│   └── shipping/          # Order fulfillment
├── components/            # React components
├── services/              # Python services
│   ├── ocr_server.py     # Text extraction
│   └── lut_processor.py  # Color grading
├── prisma/                # Database schema
├── luts/                  # Color grading files
├── uploads/               # Temporary uploads
└── processed/             # Processed images
```

## Next Steps

1. **Process 10-20 test books** to get comfortable with workflow
2. **Customize LUTs** if needed for your specific photography style
3. **Set up marketplace accounts** (AbeBooks, Biblio, etc.)
4. **Configure shipping presets** for common package sizes
5. **Train any employees** on keyboard shortcuts

## Getting Help

### Logs Location:
- Next.js: Terminal running `npm run dev`
- OCR Service: Terminal running `ocr_server.py`
- Database: `./data/postgres/pg_log/`

### Common Commands:
```bash
# Restart everything
npm run db:stop && npm run db:start
# Kill all Node processes and restart

# Reset database (warning: deletes all data)
npm run db:reset

# Update dependencies
npm update
pip install -r requirements.txt --upgrade
```

## Moving to Production

Once tested locally, you can deploy to:

1. **Vercel** (recommended for Next.js): 
   - Push to GitHub
   - Connect Vercel
   - Add environment variables

2. **Supabase** (recommended for database):
   - Export local schema
   - Create Supabase project
   - Update DATABASE_URL

3. **Railway/Render** (for Python services):
   - Deploy OCR and LUT services
   - Update service URLs

## Success Checklist

- [ ] System starts without errors
- [ ] Can upload images
- [ ] OCR extracts text successfully
- [ ] Metadata appears from APIs
- [ ] LUT processing works
- [ ] Can review and edit books
- [ ] Export generates valid CSVs
- [ ] Keyboard shortcuts work

---

**Ready to process your first batch!**

For questions or issues, check the detailed README.md or logs.
