#test my here docker run -v ${PWD}:/home -it kalilinux/kali-rolling
#docker run -it cselab/korali:latest
#docker run -it cselab/korali:3.0.2
# Destination in korali
#PWDD=/home/ubuntu/korali/examples/reinforcement.learning/flowControl_turb
PWDD=/home/ubuntu/korali/flowControl_turb

# Home
PWDH=$(pwd)
#PWDH=/home/exouser/mount/docker/flowControl_turb
#PWDH=/media/volume/sdb/docker/flowControl_turb

containername=$(basename $OLDPWD)
echo $containername

#docker run -v ${PWDH}:${PWDD} -it cselab/korali:3.0.2
#docker run -v ${PWDH}:${PWDD} -it cselab/korali:latest
docker run --name $containername -v ${PWDH}:${PWDD} -it cselab/korali:latest
#docker start -a -i flamboyant_noether
