from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='hello_world_dag',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag:
    
    hello_task = BashOperator(
        task_id='echo_hello_world',
        bash_command='echo hello world',
    )
