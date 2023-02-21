scipio
======

*scipio* refers to Cicero's *Dream of Scipio*, which ["describes an ascent through the celestial spheres"](https://en.wikipedia.org/wiki/Celestial_spheres#Literary_and_visual_expressions). This library performs a simple forward and backward spherical harmonic expansion. It leans heavily on GSL and FFTW; see the citations below.

```bib
@book{gsl271,
       author = {Galassi, Mark and Davies, Jim and Theiler, James and Gough, Brian and Priedhorsky, Reid and Jungman, Gerard and Booth, Michael and Rossi, Fabrice and Piccardi, Simone and Perassi, Carlo and Dan, Ho-Jin and Jaroszewicz, Szymon and Darnis, Nicolas and Keskitalo, Tuomo and Alxneit, Ivo and Stover, Jason~H. and Alken, Patrick and Ulerich, Rhys and Holoborodko, Pavel and Gonnet, Pedro},
        title = {GNU Scientific Library Reference Manual},
      edition = {3rd},
         isbn = {0954612078},
 howpublished = {http://www.gnu.org/software/gsl}
}

@article{fftw05,
  author = {Frigo, Matteo and Johnson, Steven~G.},
   title = {The Design and Implementation of {FFTW3}},
 journal = {Proceedings of the IEEE},
    year = 2005,
  volume = 93,
  number = 2,
   pages = {216--231},
    note = {Special issue on ``Program Generation, Optimization, and Platform Adaptation''}
}
```

To include *scipio* into a larger project, simply clone it as a submodle
with the command below.

```sh
git submodule add https://github.com/emprice/scipio.git
cmake -DGSL_ROOT=/path/to/gsl -G Ninja -B build .
```

Write some code that uses the interface. A minimal working example is
given below.

```cpp
#include <scipio.hh>

#include <cmath>
#include <iomanip>
#include <gsl/gsl_sf_legendre.h>

using namespace scipio;

int main(int argc, char *argv[])
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
```

In the `CMakeLists.txt` for the bigger project, add the submodule
directory and link as you would any other library.

```cmake
add_subdirectory(scipio)
add_executable(hello hello.cc)
target_link_libraries(hello PRIVATE scipio)
```

<!-- vim: set ft=markdown: -->
