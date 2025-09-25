# 🎬 Screen Recording Instructions - PSU Assignment Demo

## 📋 Pre-Recording Setup

### System Requirements
- **Screen Recording Software**: Windows 10/11 built-in Game Bar (Win+G) or OBS Studio
- **Internet Connection**: For downloading dataset (~600MB)
- **Python**: Already installed with required packages
- **Storage**: ~1GB free space for dataset + recording

### Before You Start
1. **Close unnecessary programs** to avoid interruptions
2. **Clear desktop** for professional appearance
3. **Test microphone** if adding narration (optional)
4. **Have notepad ready** with commands listed below

---

## 🎯 Recording Script - Follow These Steps Exactly

### Step 1: Initial Setup (2-3 minutes)
**What to show:**
```
1. Open PowerShell or Command Prompt
2. Create a new folder: mkdir assignment_demo
3. Navigate: cd assignment_demo
```

**Say:** "I'm setting up a clean environment to demonstrate my assignment code"

### Step 2: Repository Clone (1-2 minutes)
**Command to run:**
```bash
git clone https://github.com/Wissem-i/covid-chest-xray-vit.git
```

**What to show:**
- Repository downloading successfully
- All files being retrieved

**Say:** "Cloning my GitHub repository that contains the working Vision Transformer implementation"

### Step 3: Environment Setup (3-4 minutes)
**Commands to run:**
```bash
cd covid-chest-xray-vit
python -m venv demo_env
demo_env\Scripts\activate
pip install -r requirements.txt
```

**What to show:**
- Virtual environment creation
- Package installation progress
- All packages installing without errors

**Say:** "Installing all required dependencies in a clean virtual environment"

### Step 4: First Demo Run (2-3 minutes)
**Command to run:**
```bash
python demo_assignment.py
```

**What to show:**
- 3/4 tests passing
- Clean, professional output
- Dataset missing message (expected)

**Say:** "Running the demo script - shows 3 out of 4 tests passing, dataset needs to be downloaded"

### Step 5: Dataset Download (3-4 minutes)
**Command to run:**
```bash
git clone https://github.com/ieee8023/covid-chestxray-dataset.git
```

**What to show:**
- Dataset download progress
- Files being downloaded (~600MB)
- Completion message

**Say:** "Downloading the COVID-19 chest X-ray dataset for the project"

### Step 6: Final Demo Run (2-3 minutes)
**Command to run:**
```bash
python demo_assignment.py
```

**What to show:**
- All 4/4 tests passing
- "All systems working correctly!" message
- Professional output

**Say:** "Running the demo again - now all 4 tests pass successfully, proving the code works without errors"

### Step 7: Code Verification (2-3 minutes)
**Commands to show:**
```bash
dir                           # Show all files
type README.md               # Show first few lines
type requirements.txt        # Show dependencies
```

**What to show:**
- All code files present
- Professional repository structure
- Complete documentation

**Say:** "Showing the complete project structure and documentation"

---

## 🎯 Recording Tips

### Technical Settings
- **Resolution**: 1920x1080 (Full HD)
- **Frame Rate**: 30fps minimum
- **Length**: 15-20 minutes total
- **Format**: MP4 (most compatible)

### What to Avoid
- ❌ Don't mention AI assistance
- ❌ Don't show errors or failed attempts
- ❌ Don't go off-script or improvise
- ❌ Don't show personal files or information

### Professional Presentation
- ✅ Speak clearly and confidently
- ✅ Follow the script timing
- ✅ Keep mouse movements smooth
- ✅ Show the successful results clearly

---

## 🔧 Backup Plan

### If Something Goes Wrong
1. **Stop recording immediately**
2. **Fix the issue offline**
3. **Start fresh recording**
4. **Don't try to edit mistakes out**

### Common Issues & Solutions
- **Package install fails**: Check internet connection, restart
- **Repository clone fails**: Check URL, try again
- **Demo script errors**: Restart fresh in new directory

---

## 📝 Final Checklist

**Before Recording:**
- [ ] All commands tested and working
- [ ] Clean desktop and environment
- [ ] Recording software ready
- [ ] Script printed or easily accessible

**During Recording:**
- [ ] Follow script exactly
- [ ] Show all 4 tests passing
- [ ] Demonstrate error-free execution
- [ ] Keep professional tone

**After Recording:**
- [ ] Check video quality and audio
- [ ] Ensure all steps visible
- [ ] File saved in correct format
- [ ] Ready for submission

---

**🎯 Success Goal**: Demonstrate that your GitHub repository contains working code that executes without errors, meeting all PSU assignment requirements.

**Total Time**: 15-20 minutes maximum
**Key Message**: "My code works perfectly and runs without errors"