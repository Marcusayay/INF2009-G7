# Connecting/SSH

ssh -i frontend_instance.pem ec2-user@3.107.13.159



## Push to EC2 Instance
sudo dnf install git -y (Do this first)
scp -i frontend_instance.pem -r dist/* ec2-user@3.107.13.159:/usr/share/nginx/html/


# Or in EC2 SSH

cd ~/INF2006-DAaaS/frontend && git pull && npm run build