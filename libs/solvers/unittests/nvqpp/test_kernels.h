#include "cudaq.h"

__qpu__ void hartreeFock2Electrons(cudaq::qvector<> &q);
__qpu__ void ansatz(std::vector<double> theta);
__qpu__ void ansatzNonStdSignature(double theta, int N);
__qpu__ void callUccsdStatePrep(std::vector<double> params);
__qpu__ void callUccsdStatePrepWithArgs(std::vector<double> params,
                                        std::size_t numQubits,
                                        std::size_t numElectrons);