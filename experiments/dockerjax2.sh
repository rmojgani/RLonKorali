#test my here docker run -v ${PWD}:/home -it kalilinux/kali-rolling
#docker run -it cselab/korali:latest
#docker run -it cselab/korali:3.0.2
# Destination in korali
#PWDD=/home/ubuntu/korali/examples/reinforcement.learning/flowControl_turb
PWDD=/home/ubuntu/korali/flowControl_turb/

# Home
#PWDH=$(pwd)
PWDH=/home/exouser/mountphy/
#PWDH=/media/volume/sdb/docker/flowControl_turb

containername=$(basename $OLDPWD)
containername=$containername$(( $RANDOM % 1000 ))
echo $containername


#docker run --name $containername -it --gpus all rmojgani/koraligpu:1.0 #cselab/korali:latest
#docker run -v ${PWDH}:${PWDD} -it rmojgani/koraligpu:1.0
#docker run -v ${pwd}:/home/ubuntu/korali/flowControl_turb -it rmojgani/koraligpu:1.0
#docker run -v ${PWDH}:${PWDD} -it cselab/korali:3.0.2
docker run --name $containername -v ${PWDH}:${PWDD} -it --gpus all rmojgani/koraligpu:1.0 #cselab/korali:latest
#docker start -a -i flamboyant_noether
~
~
