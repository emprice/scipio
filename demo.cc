#include <scipio.hh>

#include <cmath>
#include <iomanip>
#include <gsl/gsl_sf_legendre.h>

using namespace scipio;

int main([[maybe_unused]] int argc, [[maybe_unused]] char *argv[])
{
    static constexpr size_t Nth = 17;
    static constexpr size_t Nphi = 8;

    static constexpr int l = 4;
    static constexpr int m = 2;

    auto mesh = SphericalMesh(Nth, Nphi);
    auto samp = mesh.makeArray();
    auto transform = YlmTransformer(mesh);

    for (size_t i = 0; i < Nth; ++i)
    {
        for (size_t j = 0; j < Nphi; ++j)
        {
            samp(i,j) = gsl_sf_legendre_Plm(l, m, mesh.theta().cos(i)) *
                cos(m * mesh.phi(j));
        }
    }

    std::cout << std::fixed << std::setprecision(4) << std::showpos;

    std::cout << samp << std::endl;
    transform.forward(samp);
    std::cout << samp << std::endl;
    transform.backward(samp);
    std::cout << samp << std::endl;

    return 0;
}

// vim: set ft=cpp.doxygen:
