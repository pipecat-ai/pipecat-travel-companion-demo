## Pipecat Travel Companion

To run these examples:

1. **Setup Virtual Environment** (recommended):

   ```bash
   cd server
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Installation**:

   Install the package in development mode:

   ```bash
   pip install -r requirements.txt
   ```
   
3. **Configuration**:

   Copy `env.example` to `.env` in the examples directory:

   ```bash
   cp env.example .env
   ```

   Add your API keys and configuration. Looking for a Daily API key and room URL? Sign up on the [Daily Dashboard](https://dashboard.daily.co).

4. **Running**:
   ```bash
   cd server
   python src/server.py --host YOUR_IP
   ```
