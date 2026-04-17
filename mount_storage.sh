#!/bin/bash

# mount the block storage to the host machine and set permissions
# only run this script once when you first set up storage
echo "Mounting block storage to /mnt/block/"
echo " Make sure to attach the block storage to the host machine before running this script."

lsblk

sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%

lsblk
echo "Make sure to see the vdb1 partition"

echo "Now formatting the partition with ext4 filesystem"
sudo mkfs.ext4 /dev/vdb1

echo "Mounting the partition to /mnt/block/"
sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block
sudo chown -R cc /mnt/block
sudo chgrp -R cc /mnt/block

# mount the object storage to the host machine and set permissions
# only run this script once when you first set up storage
echo "Mounting object storage to /mnt/object/"
echo " Make sure to set up the object storage and get the access key, secret key, and endpoint before running this script."

# install rclone if not already installed
if ! command -v rclone &> /dev/null
then
    echo "rclone could not be found, installing rclone..."
    curl https://rclone.org/install.sh | sudo bash
fi

sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

mkdir -p ~/.config/rclone

echo "Now configuring rclone for the object storage, MAKE SURE TO REPLACE THE PLACEHOLDERS WITH YOUR ACTUAL ACCESS KEY, SECRET KEY."
cat <<EOF > ~/.config/rclone/rclone.conf
[rclone_s3]
type = s3
provider = Ceph
access_key_id = ACCESS_KEY
secret_access_key = SECRET_ACCESS_KEY
endpoint = https://chi.tacc.chameleoncloud.org:7480
EOF


echo "test if the rclone can talk to s3"
rclone lsd rclone_s3:

echo "Mounting the object storage to /mnt/object/"
sudo mkdir -p /mnt/object
sudo chown -R cc /mnt/object
sudo chgrp -R cc /mnt/object

rclone mount rclone_s3:ObjStore_proj25 /mnt/object \
  --allow-other \
  --vfs-cache-mode off \
  --dir-cache-time 10s \
  --daemon

echo "Done mounting the block storage and object storage. You can now run the data pipeline with the mounted storages."