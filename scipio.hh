/**
 * @file scipio.hh
 * @brief Simple library for spherical harmonic decomposition; see the README
 * for the applicable symmetry assumptions
 */

#ifndef SCIPIO_HH
#define SCIPIO_HH

#include <fftw3.h>

#include <memory>
#include <iostream>
#include <functional>

namespace scipio
{

/**
 * @brief Wrapper for an array of theta (altitudinal angle) nodes, weights,
 * and related pre-computed quantities. For the purposes of integration,
 * the nodes are chosen to be the arccosine of the Gauss-Legendre nodes on the
 * interval [-1,1], so the nodes cover the domain [0,pi].
 */
struct ThetaArray
{
    /**
     * @brief Constructs and fills the array
     * @param[in] nth Number of desired theta nodes
     */
    ThetaArray(size_t nth);

    /// Copy construction is explicitly disallowed
    ThetaArray(ThetaArray const& other) = delete;

    /// Copy assignment is explicitly disallowed
    ThetaArray& operator=(ThetaArray const& other) = delete;

    /// Move constructor; swaps an existing instance with another
    ThetaArray(ThetaArray&& other) = default;

    /// Move assignment; swaps an existing instance with another
    ThetaArray& operator=(ThetaArray&& other) = default;

    /// Returns the number of nodes
    size_t size() const;

    /// Returns a (const) pointer to the nodes
    const double *data() const;

    /// Retrieves the ith node value
    double node(size_t i) const;

    /// Retrieves the ith weight value
    double weight(size_t i) const;

    /// Returns the precomputed sine of the ith node
    double sin(size_t i) const;

    /// Returns the precomputed cosine of the ith node
    double cos(size_t i) const;

    private:
        size_t m_nth;                           ///< Number of nodes

        std::unique_ptr<double[]> m_nodes;      ///< Node data
        std::unique_ptr<double[]> m_weights;    ///< Weight data

        std::unique_ptr<double[]> m_sin;        ///< Sine data
        std::unique_ptr<double[]> m_cos;        ///< Cosine data
};

/**
 * @brief Wrapper for an array of phi (azimuthal angle) nodes and related
 * pre-computed quantities. For the purposes of integration, the nodes are
 * chosen to be equidistant on the interval [0,pi/2]. The other three
 * "quadrants" of phi are redundant and so are not stored.
 */
struct PhiArray
{
    /**
     * @brief Constructs and fills the array
     * @param[in] nphi Number of desired phi nodes
     */
    PhiArray(size_t nphi);

    /// Copy construction is explicitly disallowed
    PhiArray(PhiArray const& other) = delete;

    /// Copy assignment is explicitly disallowed
    PhiArray& operator=(PhiArray const& other) = delete;

    /// Move constructor; swaps an existing instance with another
    PhiArray(PhiArray&& other) = default;

    /// Move assignment; swaps an existing instance with another
    PhiArray& operator=(PhiArray&& other) = default;

    /// Returns the number of nodes
    size_t size() const;

    /// Returns a (const) pointer to the nodes
    const double *data() const;

    /// Retrieves the jth node value
    double node(size_t j) const;

    /// Returns the precomputed sine of the jth node
    double sin(size_t j) const;

    /// Returns the precomputed cosine of the jth node
    double cos(size_t j) const;

    private:
        size_t m_nphi;                      ///< Number of nodes

        std::unique_ptr<double[]> m_nodes;  ///< Node data

        std::unique_ptr<double[]> m_sin;    ///< Sine data
        std::unique_ptr<double[]> m_cos;    ///< Cosine data
};

/// Wrapper for a contiguously-stored, 2-dimensional array
struct DataArray
{
    DataArray();
    DataArray(size_t nth, size_t nphi);
    DataArray(size_t nth, size_t nphi, double *ptr);
    ~DataArray();

    /// Copy construction is explicitly disallowed
    DataArray(DataArray const& other) = delete;

    /// Copy assignment is explicitly disallowed
    DataArray& operator=(DataArray const& other) = delete;

    /// Move constructor; swaps an existing instance with another
    DataArray(DataArray&& other);

    /// Move assignment; swaps an existing instance with another
    DataArray& operator=(DataArray&& other);

    /// Returns the number of values in the first (theta) dimension
    size_t nth() const;

    /// Returns the number of values in the second (phi) dimension
    size_t nphi() const;

    /// Returns the total number of values
    size_t size() const;

    /// Returns a read-only pointer to the data array
    const double *data() const;

    /// Returns a pointer to the data array
    double *data();

    /// Accesses the (i,j) element of the array immutably
    double operator()(size_t i, size_t j) const;

    /// Accesses the (i,j) element of the array mutably
    double& operator()(size_t i, size_t j);

    friend std::ostream& operator<<(std::ostream& os, DataArray const& arr);

    private:
        size_t m_nth;       ///< Number of theta-dimension values
        size_t m_nphi;      ///< Number of phi-dimension values
        double *m_data;     ///< Raw data storage
        bool m_owndata;     ///< Flag for data ownership

        size_t index(size_t i, size_t j) const;
};

/// Prints a nicely-formatted representation of the array to a stream
std::ostream& operator<<(std::ostream& os, DataArray const& arr);

/// Encapsulates an outer product mesh in theta and phi
struct SphericalMesh
{
    /**
     * @brief Constructs and fills the mesh
     * @param[in] nth Number of theta nodes
     * @param[in] nphi Number of phi nodes
     */
    SphericalMesh(size_t nth, size_t nphi);

    /// Copy construction is explicitly disallowed
    SphericalMesh(SphericalMesh const& other) = delete;

    /// Copy assignment is explicitly disallowed
    SphericalMesh& operator=(SphericalMesh const& other) = delete;

    /// Move constructor; swaps an existing instance with another
    SphericalMesh(SphericalMesh&& other) = default;

    /// Move assignment; swaps an existing instance with another
    SphericalMesh& operator=(SphericalMesh&& other) = default;

    /// Allows read-only access to the underlying theta array
    ThetaArray const& theta() const;

    /// Allows read-only access to the underlying phi array
    PhiArray const& phi() const;

    /// Returns the ith theta node value
    double theta(size_t i) const;

    /// Returns the jth phi node value
    double phi(size_t j) const;

    /// Creates a data array compatible with this mesh
    DataArray makeArray() const;

    /// Creates a wrapper data array compatible with this mesh
    DataArray wrapArray(double *ptr) const;

    private:
        size_t m_nth;   ///< Number of theta nodes
        size_t m_nphi;  ///< Number of phi nodes

        ThetaArray m_theta;     ///< Array of theta values
        PhiArray m_phi;         ///< Array of phi values
};

/**
 * @brief Lookup table for compactly-stored, unique values of the
 * associated Legendre polynomials on a given mesh
 */
struct YlmLookupTable
{
    /// Constructs and fills the lookup table
    YlmLookupTable(SphericalMesh const& mesh);

    /// Copy construction is explicitly disallowed
    YlmLookupTable(YlmLookupTable const& other) = delete;

    /// Copy assignment is explicitly disallowed
    YlmLookupTable& operator=(YlmLookupTable const& other) = delete;

    /// Move constructor; swaps an existing instance with another
    YlmLookupTable(YlmLookupTable&& other) = default;

    /// Move assignment; swaps an existing instance with another
    YlmLookupTable& operator=(YlmLookupTable&& other) = default;

    /// Returns the maximum value of ell used in constructing the table
    size_t lmax() const;

    /**
     * @brief Returns a value from the table
     * @param[in] i Index of theta in the mesh
     * @param[in] l Value of ell for the polynomial
     * @param[in] m Value of em for the polynomial
     * @return Value of \f$ P_\ell^m\!\left(\theta_i\right) \f$
     */
    double operator()(size_t i, size_t l, size_t m) const;

    private:
        size_t m_nth;   ///< Number of theta values in the table
        size_t m_lmax;  ///< Maximum value of ell in the table

        std::unique_ptr<double[]> m_data;   ///< Raw table data

        /// Helper function for computing indices into @c m_data
        size_t index(size_t i, size_t l, size_t m) const;
};

/// Computes the forward and backward spherical harmonic transforms
struct YlmTransformer
{
    YlmTransformer(SphericalMesh const& mesh);
    ~YlmTransformer();

    /// Copy construction is explicitly disallowed
    YlmTransformer(YlmTransformer const& other) = delete;

    /// Copy assignment is explicitly disallowed
    YlmTransformer& operator=(YlmTransformer const& other) = delete;

    /// Move constructor; swaps an existing instance with another
    YlmTransformer(YlmTransformer&& other);

    /// Move assignment; swaps an existing instance with another
    YlmTransformer& operator=(YlmTransformer&& other);

    /// Computes the forward transform, overwriting the input
    void forward(DataArray& inout);

    /// Computes the backward transform, overwriting the input
    void backward(DataArray& inout);

    /// Computes the forward transform, preserving the input
    void forward(DataArray const& in, DataArray& out);

    /// Computes the backward transform, preserving the input
    void backward(DataArray const& in, DataArray& out);

    private:
        /// FFT plan that does not necessarily preserve its input
        fftw_plan m_inplace_plan;

        /// FFT plan that must preserve its input
        fftw_plan m_outofplace_plan;

        /// Copyable reference to a mesh
        std::reference_wrapper<const SphericalMesh> m_mesh;

        size_t m_nth;   ///< Number of theta mesh points
        size_t m_nphi;  ///< Number of phi mesh points

        YlmLookupTable m_lookup;    ///< Polynomial lookup table
        size_t m_lmax;              ///< Maximum value of ell in the table

        DataArray m_wk;     ///< Scratch data array
};

}   // namespace scipio

#endif      // SCIPIO_HH

// vim: set ft=cpp.doxygen:
