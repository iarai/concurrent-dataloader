``` 
zip -r storage-benchmarking.zip storage-benchmarking/
scp -i ~/iarai-storage-benchmarking-christian.pem storage-benchmarking.zip ec2-user@ec2-34-252-214-174.eu-west-1.compute.amazonaws.com:/home/ec2-user/
ssh -i ~/iarai-storage-benchmarking-christian.pem ec2-user@ec2-34-252-214-174.eu-west-1.compute.amazonaws.com unzip storage-benchmarking.zip
ssh -i ~/iarai-storage-benchmarking-christian.pem ec2-user@ec2-34-252-214-174.eu-west-1.compute.amazonaws.com python3 -m pip install boto3 numpy==1.21.1 ray==1.6.0
ssh -i ~/iarai-storage-benchmarking-christian.pem ec2-user@ec2-34-252-214-174.eu-west-1.compute.amazonaws.com


[ec2-user@ip-172-31-249-175 ~]$ export AWS_ACCESS_KEY_ID=XXX; export AWS_SECRET_ACCESS_KEY=XXX; cd storage-benchmarking/src; printenv; python3 ../boto3_threadpool.py
[ec2-user@ip-172-31-249-175 src]$ cd 
[ec2-user@ip-172-31-249-175 ~]$ export AWS_ACCESS_KEY_ID=XXX; export AWS_SECRET_ACCESS_KEY=XXX/3RI; cd storage-benchmarking; printenv; python3 ray_.py
```
