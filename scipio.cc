#include "scipio.hh"

#include <cmath>
#include <cassert>
#include <utility>
#include <algorithm>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_legendre.h>

using namespace scipio;

ThetaArray::ThetaArray(size_t nth) : m_nth(nth)
{
    m_nodes   = new double[m_nth];
    m_weights = new double[m_nth];

    m_sin = new double[m_nth];
    m_cos = new double[m_nth];

    // get the quadrature nodes and weights
    gsl_integration_fixed_workspace *wk =
        gsl_integration_fixed_alloc(gsl_integration_fixed_legendre,
            m_nth, -1, 1, 0, 0);
    std::copy(wk->x, wk->x + m_nth, m_cos);
    std::copy(wk->weights, wk->weights + m_nth, m_weights);
    gsl_integration_fixed_free(wk);

    // precompute the values of theta = acos(x) and sin(theta)
    for (size_t i = 0; i < m_nth; ++i)
    {
        m_nodes[i] = std::acos(m_cos[i]);
        m_sin[i] = std::sin(m_nodes[i]);
    }
}

ThetaArray::~ThetaArray()
{
    delete[] m_nodes;
    delete[] m_weights;

    delete[] m_sin;
    delete[] m_cos;
}

ThetaArray::ThetaArray(ThetaArray&& other)
{
    std::swap(m_nth, other.m_nth);
    std::swap(m_nodes, other.m_nodes);
    std::swap(m_weights, other.m_weights);

    std::swap(m_sin, other.m_sin);
    std::swap(m_cos, other.m_cos);
}

ThetaArray& ThetaArray::operator=(ThetaArray&& other)
{
    std::swap(m_nth, other.m_nth);
    std::swap(m_nodes, other.m_nodes);
    std::swap(m_weights, other.m_weights);

    std::swap(m_sin, other.m_sin);
    std::swap(m_cos, other.m_cos);

    return *this;
}

size_t ThetaArray::size() const { return m_nth; }

const double *ThetaArray::data() const { return m_nodes; }

double ThetaArray::node(size_t i) const { return m_nodes[i]; }

double ThetaArray::weight(size_t i) const { return m_weights[i]; }

double ThetaArray::sin(size_t i) const { return m_sin[i]; }

double ThetaArray::cos(size_t i) const { return m_cos[i]; }

PhiArray::PhiArray(size_t nphi) : m_nphi(nphi)
{
    m_nodes = new double[m_nphi];

    m_sin = new double[m_nphi];
    m_cos = new double[m_nphi];

    double dphi = M_PI_2 / (m_nphi - 1);

    // precompute the values of phi, cos(phi), and sin(phi)
    for (size_t j = 0; j < m_nphi; ++j)
    {
        m_nodes[j] = j * dphi;
        sincos(m_nodes[j], m_sin + j, m_cos + j);
    }
}

PhiArray::~PhiArray()
{
    delete[] m_nodes;

    delete[] m_sin;
    delete[] m_cos;
}

PhiArray::PhiArray(PhiArray&& other)
{
    std::swap(m_nphi, other.m_nphi);
    std::swap(m_nodes, other.m_nodes);

    std::swap(m_sin, other.m_sin);
    std::swap(m_cos, other.m_cos);
}

PhiArray& PhiArray::operator=(PhiArray&& other)
{
    std::swap(m_nphi, other.m_nphi);
    std::swap(m_nodes, other.m_nodes);

    std::swap(m_sin, other.m_sin);
    std::swap(m_cos, other.m_cos);

    return *this;
}

size_t PhiArray::size() const { return m_nphi; }

const double *PhiArray::data() const { return m_nodes; }

double PhiArray::node(size_t j) const { return m_nodes[j]; }

double PhiArray::sin(size_t j) const { return m_sin[j]; }

double PhiArray::cos(size_t j) const { return m_cos[j]; }

DataArray::DataArray(size_t nth, size_t nphi) :
    m_nth(nth), m_nphi(nphi)
{
    // allocate aligned memory for fftw
    m_data = reinterpret_cast<double*>
        (fftw_malloc(nth * nphi * sizeof(double)));
}

DataArray::~DataArray()
{
    // free the aligned memory
    if (m_data) fftw_free(m_data);
}

DataArray::DataArray(DataArray&& other) : m_data(nullptr)
{
    std::swap(m_nth, other.m_nth);
    std::swap(m_nphi, other.m_nphi);
    std::swap(m_data, other.m_data);
}

DataArray& DataArray::operator=(DataArray&& other)
{
    std::swap(m_nth, other.m_nth);
    std::swap(m_nphi, other.m_nphi);
    std::swap(m_data, other.m_data);

    return *this;
}

size_t DataArray::nth() const { return m_nth; }

size_t DataArray::nphi() const { return m_nphi; }

const double *DataArray::data() const { return m_data; }

double *DataArray::data() { return m_data; }

double DataArray::operator()(size_t i, size_t j) const
{
    return m_data[index(i,j)];
}

double& DataArray::operator()(size_t i, size_t j)
{
    return m_data[index(i,j)];
}

size_t DataArray::index(size_t i, size_t j) const { return i * m_nphi + j; }

std::ostream& scipio::operator<<(std::ostream& os, DataArray const& arr)
{
    for (size_t i = 0; i < arr.nth(); ++i)
    {
        // row number at beginning of line
        os << "(" << i << ") ";

        for (size_t j = 0; j < arr.nphi(); ++j)
        {
            // space-delimited data values on this row
            os << arr(i,j) << " ";
        }

        os << std::endl;
    }

    return os;
}

SphericalMesh::SphericalMesh(size_t nth, size_t nphi) :
    m_nth(nth), m_nphi(nphi), m_theta(m_nth), m_phi(m_nphi)
{
    // no-op
}

ThetaArray const& SphericalMesh::theta() const { return m_theta; }

PhiArray const& SphericalMesh::phi() const { return m_phi; }

double SphericalMesh::theta(size_t i) const { return m_theta.node(i); }

double SphericalMesh::phi(size_t j) const { return m_phi.node(j); }

DataArray SphericalMesh::makeArray() const
{
    return DataArray(m_nth, m_nphi);
}

YlmLookupTable::YlmLookupTable(SphericalMesh const& mesh) :
    m_nth(mesh.theta().size()), m_lmax(m_nth - 1)
{
    size_t nsz = m_nth * (m_lmax + 2) * (m_lmax + 2) / 4;
    m_data = new double[nsz];

    double *lwk = new double[gsl_sf_legendre_array_n(m_lmax)];

    for (size_t i = 0; i < m_nth; ++i)
    {
        // fill array of legendre polynomial values at fixed angle
        gsl_sf_legendre_array(GSL_SF_LEGENDRE_SPHARM, m_lmax,
            mesh.theta().cos(i), lwk);

        // store the values in the data layout we expect
        for (size_t l = 0; l <= m_lmax; ++l)
        {
            for (size_t m = 0; m <= l; m += 2)
            {
                size_t idx = index(i, l, m);
                assert(idx < nsz);
                m_data[idx] = lwk[gsl_sf_legendre_array_index(l, m)];
            }
        }
    }

    delete[] lwk;
}

YlmLookupTable::~YlmLookupTable()
{
    if (m_data) delete[] m_data;
}

YlmLookupTable::YlmLookupTable(YlmLookupTable&& other) : m_data(nullptr)
{
    std::swap(m_nth, other.m_nth);
    std::swap(m_data, other.m_data);
}

YlmLookupTable& YlmLookupTable::operator=(YlmLookupTable&& other)
{
    std::swap(m_nth, other.m_nth);
    std::swap(m_data, other.m_data);

    return *this;
}

size_t YlmLookupTable::lmax() const { return m_lmax; }

double YlmLookupTable::operator()(size_t i, size_t l, size_t m) const
{
    return m_data[index(i,l,m)];
}

size_t YlmLookupTable::index(size_t i, size_t l, size_t m) const
{
    assert((i < m_nth) && (m <= l));
    size_t offset = ((l % 2) ? ((l + 1) * (l + 1)) : (l * (l + 2))) / 4;
    return (offset + (m / 2)) * m_nth + i;
}

YlmTransformer::YlmTransformer(SphericalMesh const& mesh) :
    m_mesh(mesh), m_nth(m_mesh.get().theta().size()),
    m_nphi(m_mesh.get().phi().size()), m_lookup(m_mesh.get()),
    m_lmax(m_lookup.lmax()), m_wk(m_mesh.get().makeArray())
{
    double *wk_in = reinterpret_cast<double*>
        (fftw_malloc(m_nth * m_nphi * sizeof(double)));
    double *wk_out = reinterpret_cast<double*>
        (fftw_malloc(m_nth * m_nphi * sizeof(double)));

    int sz[] = { m_nphi };
    fftw_r2r_kind kinds[] = { FFTW_REDFT00 };

    // plan an in-place fft that may overwrite its input
    m_inplace_plan = fftw_plan_many_r2r(1, sz, m_nth, wk_in, sz,
        1, m_nphi, wk_out, sz, 1, m_nphi, kinds, FFTW_PATIENT);

    // plan an out-of-place fft that may not overwrite its input
    m_outofplace_plan = fftw_plan_many_r2r(1, sz, m_nth, wk_in, sz,
        1, m_nphi, wk_out, sz, 1, m_nphi, kinds,
        FFTW_PATIENT | FFTW_PRESERVE_INPUT);

    fftw_free(wk_in);
    fftw_free(wk_out);
}

YlmTransformer::~YlmTransformer()
{
    if (m_inplace_plan) fftw_destroy_plan(m_inplace_plan);
    if (m_outofplace_plan) fftw_destroy_plan(m_outofplace_plan);
}

YlmTransformer::YlmTransformer(YlmTransformer&& other) :
    m_inplace_plan(nullptr), m_outofplace_plan(nullptr), m_mesh(other.m_mesh),
    m_lookup(std::move(other.m_lookup)), m_wk(std::move(other.m_wk))
{
    std::swap(m_inplace_plan, other.m_inplace_plan);
    std::swap(m_outofplace_plan, other.m_outofplace_plan);

    std::swap(m_nth, other.m_nth);
    std::swap(m_nphi, other.m_nphi);
    std::swap(m_lmax, other.m_lmax);
}

YlmTransformer& YlmTransformer::operator=(YlmTransformer&& other)
{
    std::swap(m_inplace_plan, other.m_inplace_plan);
    std::swap(m_outofplace_plan, other.m_outofplace_plan);

    std::swap(m_mesh, other.m_mesh);
    std::swap(m_nth, other.m_nth);
    std::swap(m_nphi, other.m_nphi);

    std::swap(m_lookup, other.m_lookup);
    std::swap(m_lmax, other.m_lmax);

    std::swap(m_wk, other.m_wk);

    return *this;
}

void YlmTransformer::forward(DataArray& inout)
{
    // compute fft to extract the cos(m phi) components
    fftw_execute_r2r(m_inplace_plan, inout.data(), m_wk.data());

    for (size_t j = 0; j < m_nphi; ++j)
    {
        size_t m = 2 * j;

        // identically zero terms because m > l
        for (size_t l = 0; l < m; ++l)
        {
            inout(l, j) = 0;
        }

        // quadrature integration sets all m <= l terms
        for (size_t l = m; l <= m_lmax; ++l)
        {
            double sum = 0;

            for (size_t i = 0; i < m_nth; ++i)
            {
                sum += m_mesh.get().theta().weight(i) *
                    m_wk(i, j) * m_lookup(i, l, m);
            }

            inout(l, j) = sum;
        }
    }
}

void YlmTransformer::backward(DataArray& inout)
{
    // this factor accounts for normalizing the fft output and the
    // integration over the polynomials
    double norm = M_PI / (m_nphi - 1);

    for (size_t j = 0; j < m_nphi; ++j)
    {
        size_t m = 2 * j;

        for (size_t i = 0; i < m_nth; ++i)
        {
            double sum = 0;

            // sum polynomials at fixed angle with the calculated coefficient
            for (size_t l = m; l <= m_lmax; ++l)
            {
                sum += inout(l, j) * m_lookup(i, l, m);
            }

            // store normalized sum
            m_wk(i, j) = sum * norm;
        }
    }

    // invert the fft calculation
    fftw_execute_r2r(m_inplace_plan, m_wk.data(), inout.data());
}

void YlmTransformer::forward(DataArray const& in, DataArray& out)
{
    // compute fft to extract the cos(m phi) components
    // NOTE: fftw won't overwrite the input by construction, but
    // we have to cast away the const-ness so that the function
    // signature will match
    fftw_execute_r2r(m_outofplace_plan,
        const_cast<double*>(in.data()), m_wk.data());

    for (size_t j = 0; j < m_nphi; ++j)
    {
        size_t m = 2 * j;

        // identically zero terms because m > l
        for (size_t l = 0; l < m; ++l)
        {
            out(l, j) = 0;
        }

        // quadrature integration sets all m <= l terms
        for (size_t l = m; l <= m_lmax; ++l)
        {
            double sum = 0;

            for (size_t i = 0; i < m_nth; ++i)
            {
                sum += m_mesh.get().theta().weight(i) *
                    m_wk(i, j) * m_lookup(i, l, m);
            }

            out(l, j) = sum;
        }
    }
}

void YlmTransformer::backward(DataArray const& in, DataArray& out)
{
    // this factor accounts for normalizing the fft output and the
    // integration over the polynomials
    double norm = M_PI / (m_nphi - 1);

    for (size_t j = 0; j < m_nphi; ++j)
    {
        size_t m = 2 * j;

        for (size_t i = 0; i < m_nth; ++i)
        {
            double sum = 0;

            // sum polynomials at fixed angle with the calculated coefficient
            for (size_t l = m; l <= m_lmax; ++l)
            {
                sum += in(l, j) * m_lookup(i, l, m);
            }

            // store normalized sum
            m_wk(i, j) = sum * norm;
        }
    }

    // invert the fft calculation
    fftw_execute_r2r(m_inplace_plan, m_wk.data(), out.data());
}

// vim: set ft=cpp.doxygen:
