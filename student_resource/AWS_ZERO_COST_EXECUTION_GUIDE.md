# 🚀 AWS Zero-Cost Execution Guide: Amazon ML Challenge 2026

A step-by-step handbook on how to safely use your **Amazon ML Challenge Promotional AWS Credits** to run high-performance cloud machines (**8 vCPUs, 16–32 GB RAM**) with **$0.00 out-of-pocket cost**.

---

## 📌 1. The Core Truth: Will You Be Charged?

> **NO, you will not pay a single penny out of your pocket**, provided you follow two simple rules:
> 1. Your Amazon Promotional Credit code is redeemed in your AWS Billing Console.
> 2. You **Stop** or **Terminate** your instance when your pipeline finishes.

### 💰 The Math Proof:
AWS deducts from your **promotional credit balance first** before touching any credit card.

| Resource | Spec | Hourly Cost | Time Needed for Pipeline | Credit Consumed | Out-of-Pocket Cost |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`c6i.2xlarge`** | 8 vCPU, 16 GB RAM | **~$0.34 / hr** | 2 – 3 hours | **~$1.02** | **$0.00** |
| **`m6i.2xlarge`** | 8 vCPU, 32 GB RAM | **~$0.38 / hr** | 2 – 3 hours | **~$1.14** | **$0.00** |
| **50 GB gp3 SSD Disk** | 50 GB storage | ~$0.08 / month | 3 days | **~$0.04** | **$0.00** |
| **Total Typical Usage** | | | **~3 hours** | **~$1.18** | **$0.00** |

Even a **$25.00** promotional credit gives you **over 70 hours** of high-speed execution.

---

## 🛡️ 2. Safety Setup: 100% Zero-Spend Guarantee

Before launching any server, set up an automatic alert so you have zero anxiety.

### Step 2.1: Redeem Your Credit
1. Open the [AWS Billing Console - Credits](https://console.aws.amazon.com/billing/home#/credits).
2. Click **Redeem Credit**.
3. Paste the Promo Code received from the Amazon ML Challenge email and click **Redeem**.
4. Confirm your credit balance is visible on the screen.

### Step 2.2: Set a $5 Budget Alert (Takes 1 Minute)
1. In the search bar at the top of AWS Console, search for **AWS Budgets**.
2. Click **Create a budget** $\rightarrow$ Select **Cost budget (Recommended)**.
3. Set the target budget amount to **$5.00**.
4. Add your email address under **Alert recipients** so AWS immediately notifies you if spending exceeds $5.
5. Click **Create budget**.

---

## 🖥️ 3. How to Launch the Right EC2 Machine

Do **not** use `t3.micro` (1 GB RAM) as it will crash with an Out-of-Memory (OOM) error on large data. Use `c6i.2xlarge` or `m6i.2xlarge`.

### Step-by-Step Launch Instructions:
1. Open the **Amazon EC2 Console** $\rightarrow$ Click **Launch Instance** (Orange button).
2. **Name:** `amazon-ml-pipeline`
3. **Application and OS Images (AMI):**
   - Choose **Ubuntu**
   - Version: **Ubuntu Server 24.04 LTS (HVM), SSD Volume Type** (64-bit x86)
4. **Instance Type:**
   - Search for **`c6i.2xlarge`** (8 vCPUs, 16 GB RAM) or **`m6i.2xlarge`** (8 vCPUs, 32 GB RAM).
5. **Key pair (login):**
   - Click **Create new key pair**.
   - Name it `amazon-ml-key`, format `.pem` (for Mac/Linux/OpenSSH) or `.ppk` (for PuTTY on older Windows).
   - Save the downloaded `.pem` file safely on your computer.
6. **Network Settings:**
   - Check **Allow SSH traffic from anywhere** (or *My IP* for extra security).
7. **Configure Storage:**
   - Change `8 GiB` to **`50 GiB`** of **`gp3`** (General Purpose SSD).
8. Click **Launch Instance**.

---

## ⚡ 4. Connecting and Running Your Code

### 4.1 Connect via SSH (Terminal / PowerShell)
On your local computer, open PowerShell or Terminal in the folder where your `.pem` file is saved:

```bash
# On Mac/Linux, set file permissions (not needed on Windows):
chmod 400 amazon-ml-key.pem

# SSH into your EC2 (replace YOUR_EC2_PUBLIC_IP with your instance's public IP):
ssh -i "amazon-ml-key.pem" ubuntu@YOUR_EC2_PUBLIC_IP
```

*(Tip: You can also use the **VS Code Remote - SSH** extension for a full graphical code editor inside the EC2 machine!)*

### 4.2 Install Dependencies on EC2 (Takes 2 Minutes)
Run these commands inside the EC2 terminal:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git tmux htop
python3 -m venv er_env
source er_env/bin/activate
pip install numpy pandas scikit-learn lightgbm tqdm
```

### 4.3 Keep Jobs Running Safely (Using `tmux`)
To ensure your script keeps running even if your laptop disconnects or closes:

```bash
# Start a new persistent session:
tmux new -s ml_job

# Run your Python pipeline:
python3 run_pipeline.py

# To detach safely: Press Ctrl+B, then D
# To re-attach anytime: tmux attach -t ml_job
```

### 4.4 Downloading Your Output Files to Your Laptop
Once the job is finished, run this from your **local computer's terminal**:

```bash
scp -i "amazon-ml-key.pem" -r ubuntu@YOUR_EC2_PUBLIC_IP:/home/ubuntu/output ./
```

---

## 🛑 5. The Golden Stopping Rule (Save Your Credits)

When your run is done and your output files (`matching_results.tsv` and `candidate_pairs.tsv`) are downloaded:

1. Go to the **EC2 Console** $\rightarrow$ **Instances**.
2. Select your instance $\rightarrow$ Click **Instance State**:
   - **Stop Instance:** Pauses compute. You can start it again anytime with your files intact. (Compute cost becomes **$0.00**; storage costs ~$0.10/month).
   - **Terminate Instance:** Permanently deletes the instance and disk when the competition is over. (Total cost becomes **$0.00** forever).

---

## 📋 Summary Checklist to Share with Teammates

- [x] **Credits Redeemed** in AWS Billing.
- [x] **$5 Budget Alert** created for safety.
- [x] Instance launched: **Ubuntu 24.04**, **`c6i.2xlarge`** (8 vCPU / 16 GB RAM), **50 GB gp3 disk**.
- [x] Scripts run inside **`tmux`** session.
- [x] Output files downloaded.
- [x] Instance **Stopped / Terminated**.
- [x] **Total Money Paid: $0.00** 🎉
