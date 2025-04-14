import openrouteservice
import os
from dotenv import load_dotenv
load_dotenv()

client = openrouteservice.Client(key=os.getenv("ORS_API_KEY"))
print(client)
