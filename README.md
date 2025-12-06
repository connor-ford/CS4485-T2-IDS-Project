# CS4485-T2-IDS-Project
make init # creates virtualenv, installs dependencies
make train-all # trains all models (will lag your computer, fair warning)
make serve # serves the app on Docker

cd into frontend 
python3 -m venv venv
source venv/bin/activate 
 # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Run the frontend
python app.py 

____________________________
new changes//
for frontend you can do:
make frontend-init
make frontend-serve 
