/*
 * @BEGIN LICENSE
 *
 * Forte: an open-source plugin to Psi4 (https://github.com/psi4/psi4)
 * that implements a variety of quantum chemistry methods for strongly
 * correlated electrons.
 *
 * Copyright (c) 2012-2024 by its authors (see COPYING, COPYING.LESSER,
 * AUTHORS).
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses/.
 *
 * @END LICENSE
 */

#pragma once

#include "psi4/lib3index/dfhelper.h"
#include "integrals.h"

namespace forte {

/// A DiskDFIntegrals class for avoiding the storage of the ThreeIntegral tensor
/// Assumes that the DFIntegrals are stored in a binary file generated by
/// DF_Helper
/// Aptei_xy are extremely slow -> Try to use three_electron_block.  Much faster
/// Reading individual elements is slow
class DISKDFIntegrals : public Psi4Integrals {
  public:
    /// Contructor of DISKDFIntegrals
    DISKDFIntegrals(std::shared_ptr<ForteOptions> options,
                    std::shared_ptr<psi::Wavefunction> ref_wfn,
                    std::shared_ptr<MOSpaceInfo> mo_space_info, IntegralSpinRestriction restricted);

    void initialize() override;
    /// aptei_xy functions are slow.  try to use three_integral_block

    // ==> Class public virtual functions <==

    double aptei_aa(size_t p, size_t q, size_t r, size_t s) override;
    double aptei_ab(size_t p, size_t q, size_t r, size_t s) override;
    double aptei_bb(size_t p, size_t q, size_t r, size_t s) override;

    /// Return the antisymmetrized alpha-alpha chunck as an ambit::Tensor
    ambit::Tensor aptei_aa_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                 const std::vector<size_t>& r,
                                 const std::vector<size_t>& s) override;
    /// Return the antisymmetrized alpha-beta chunck as an ambit::Tensor
    ambit::Tensor aptei_ab_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                 const std::vector<size_t>& r,
                                 const std::vector<size_t>& s) override;
    /// Return the antisymmetrized beta-beta chunck as an ambit::Tensor
    ambit::Tensor aptei_bb_block(const std::vector<size_t>& p, const std::vector<size_t>& q,
                                 const std::vector<size_t>& r,
                                 const std::vector<size_t>& s) override;

    double** three_integral_pointer() override;
    /// Read a block of the DFIntegrals and return an Ambit tensor of size A by p by q
    ambit::Tensor three_integral_block(const std::vector<size_t>& A, const std::vector<size_t>& p,
                                       const std::vector<size_t>& q,
                                       ThreeIntsBlockOrder order = Qpq) override;
    /// return ambit tensor of size A by q
    ambit::Tensor three_integral_block_two_index(const std::vector<size_t>& A, size_t p,
                                                 const std::vector<size_t>& q) override;

    void set_tei(size_t p, size_t q, size_t r, size_t s, double value, bool alpha1,
                 bool alpha2) override;

    /// Make a Fock matrix computed with respect to a given determinant
    size_t nthree() const override;

  private:
    // ==> Class data <==

    std::shared_ptr<psi::DFHelper> df_;
    std::shared_ptr<psi::Matrix> ThreeIntegral_;
    size_t nthree_ = 0;

    // ==> Class private virtual functions <==

    void gather_integrals() override;
    void resort_integrals_after_freezing() override;
};

} // namespace forte
