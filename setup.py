import json
import random
from time import sleep

import pymongo as pymongo
from sshtunnel import SSHTunnelForwarder

local_address = '0.0.0.0'
port = random.randint(10000, 15000)

with SSHTunnelForwarder(
        ("198.199.77.214", 22),
        ssh_username="sysadmin",
        ssh_pkey="/home/cmennen/.ssh/hydra_rsa",
        remote_bind_address=("localhost", 27017),
        local_bind_address=(local_address, port)
) as _:
    sleep(1)

    with pymongo.MongoClient(local_address, port=port) as client:
        sleep(1)
        db = client['tests']
        c = db['tests']
        x = json.load(open("test.json", "r"))
        for series in x:
            for test in series:
                c.insert_one({
                    "test": test,
                    "tested": False,
                    "learning_rate": {
                        "1": None,
                        "2": None,
                        "3": None,
                        "4": None,
                        "5": None,
                    }
                })
