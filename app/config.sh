#
# Example environment setup file for Fabric Engine
# Update FABRIC_DIR to point to the chosen installation dir if you
# encounter issues
#

echo "Setting up Fabric Engine 1.15.2 environment:"

FABRIC_DIR=/Volumes/MIKOO_Backup/JULIEN/Dev/FabricEngine-1.15.2-Darwin-x86_64
#$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
export FABRIC_DIR
echo "  Set FABRIC_DIR=\"$FABRIC_DIR\""

PATH=$FABRIC_DIR/bin:$PATH
export PATH
echo "  Set PATH=\"$PATH\""

PYTHON_VERSION=$(python -c 'import sys; print "%u.%u" % sys.version_info[:2]')
PYTHONPATH=$FABRIC_DIR/Python/$PYTHON_VERSION:$PYTHONPATH
export PYTHONPATH
echo "  Set PYTHONPATH=\"$PYTHONPATH\""

FABRIC_EXTS_PATH=$FABRIC_DIR/Exts
KLML_EXTS_CPP_MNIST_PATH=/Volumes/MIKOO_Backup/JULIEN/Dev/KL_ML/core/c++/MNIST
KLML_EXTS_PATH=/Volumes/MIKOO_Backup/JULIEN/Dev/KL_ML/core/kl

KLML_EXTS_CPP_PATH=$KLML_EXTS_CPP_MNIST_PATH
FABRIC_EXTS_PATH=$FABRIC_EXTS_PATH:$KLML_EXTS_CPP_PATH:$KLML_EXTS_PATH
export FABRIC_EXTS_PATH
echo "  Set FABRIC_EXTS_PATH=\"$FABRIC_EXTS_PATH\""

#if [ -n "$MAYA_VERSION" ]
#then
#  echo "  Found MAYA_VERSION=\"$MAYA_VERSION\""
#  for d in "$FABRIC_DIR/SpliceIntegrations/FabricSpliceMaya$MAYA_VERSION"*
#  do
#    MAYA_MODULE_PATH="$d:$MAYA_MODULE_PATH"
#    export MAYA_MODULE_PATH
#    echo "    Set MAYA_MODULE_PATH=\"$MAYA_MODULE_PATH\""
# done
#fi

echo "Done!"
