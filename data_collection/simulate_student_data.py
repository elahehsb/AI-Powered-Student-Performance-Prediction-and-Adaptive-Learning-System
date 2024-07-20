import time
import json
import random
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def generate_student_data():
    while True:
        data = {
            'student_id': random.randint(1, 1000),
            'timestamp': int(time.time()),
            'activity': random.choice(['lecture', 'quiz', 'assignment']),
            'content_id': random.randint(1, 500),
            'score': round(random.uniform(0, 100), 2)
        }
        producer.send('student_data', value=data)
        time.sleep(2)

generate_student_data()
