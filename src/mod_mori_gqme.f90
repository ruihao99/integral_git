! simple code used to integrate the Mori Generalized Quantum Master Equation
! Created by: Rui-Hao Bi

! problem statement:
! Given a precomputed memory kernel K(t) and 
! a frequency matrix (or first-order moment) Omega,
! compute the autocorrelation function C
! dC/dt = Omega C + \int_0^t dt' K(t') C(t-t').
! The initial condition is C(0) = Identity(dim)

!--------------------------------------------------------------------------------
module mod_mori_gqme
    use, intrinsic :: iso_fortran_env, only: dp => real64
    implicit none
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! Module variables and data !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ! - system size
    integer :: dim, nmem

    ! - output
    real(dp) :: tf, dt
    integer :: nsteps

    ! - input data
    real(dp), allocatable :: t_mem(:)
    complex(dp), allocatable :: K_mem(:,:,:)
    complex(dp), allocatable :: Omega(:,:)

    ! - dynamic variable
    integer :: istep
    complex(dp), allocatable :: C(:,:)

    ! - temporary variables
    complex(dp), allocatable :: dC_conv(:,:)
    complex(dp), allocatable :: k1(:,:), k2(:,:), k3(:,:), k4(:,:)

    ! - output data
    complex(dp), allocatable :: C_history(:,:,:)

    ! - initialization flag
    logical :: initialized = .false.

    ! - constant parametersA
    ! real(dp), parameter :: mem_tol = 1.0e-10_dp
    real(dp), parameter :: mem_tol = 1.0e-6_dp

contains
!--------------------------------------------------------------------------------
subroutine init_module(t_mem_, K_mem_, Omega_, dim_, nmem_, tf_)
    implicit none
    ! input arguments
    real(dp), intent(in) :: t_mem_(nmem_)
    complex(dp), intent(in) :: K_mem_(nmem_,dim_,dim_)
    complex(dp), intent(in) :: Omega_(dim_,dim_)
    integer, intent(in) :: dim_, nmem_
    real(dp), intent(in) :: tf_

    ! local variables
    integer :: i
    real(dp) :: norm_K_mem_0, norm_K_mem_n

    ! get the size of the memory kernel that is needed 
    do i = 1, 4
        print *, K_mem_(2, :, i)
    end do

    norm_K_mem_0 = euclidean_norm(K_mem_(1, :, :))
    ! print *, "Norm of the first memory kernel: ", norm_K_mem_0
    nmem = nmem_
    do i = 1, nmem_
        norm_K_mem_n = euclidean_norm(K_mem_(i, :, :))
        ! print *, "rel at ", i, "is: ", norm_K_mem_n / norm_K_mem_0, " tol is: ", mem_tol
        if (norm_K_mem_n < mem_tol * norm_K_mem_0) then
            nmem = i - 1
            exit
        end if
    end do
    print *, "Memory kernel size: ", nmem
    print *, "provided memory kernel size: ", nmem_


    ! allocate and initialize module variables
    dim = dim_
    nmem = nmem_
    dt = t_mem_(2) - t_mem_(1)
    tf = tf_
    nsteps = ceiling(tf / dt)
    allocate(t_mem(nmem))
    allocate(K_mem(nmem, dim, dim))
    allocate(Omega(dim, dim))
    allocate(C(dim, dim))
    allocate(dC_conv(dim, dim))
    allocate(k1(dim, dim), k2(dim, dim), k3(dim, dim), k4(dim, dim))
    allocate(C_history(nsteps, dim, dim))
    t_mem = t_mem_
    K_mem = K_mem_
    Omega = Omega_

    ! initialize the initial condition
    C = 0.0_dp
    do i = 1, dim
        C(i, i) = 1.0_dp
    end do

    ! setting the initial step
    istep = 1
    C_history(istep, :, :) = C
    initialized = .true.
end

subroutine eval_convolution(C_history_, K_mem_, dC_conv_)
    implicit none
    complex(dp), intent(in) :: C_history_(:,:,:)
    complex(dp), intent(in) :: K_mem_(:,:,:)
    complex(dp), intent(out) :: dC_conv_(:,:)
    integer :: step_now

    integer :: mem_size, his_size, upper_bound
    integer :: i

    ! initialize dC_conv_ to zero
    dC_conv_(:,:) = (0.0_dp, 0.0_dp)

    ! get the memory size
    ! mem_size = size(C_history_, 1)
    mem_size = size(K_mem_, 1)
    his_size = size(C_history_, 1)
    upper_bound = min(mem_size, his_size)

    do i = 1, upper_bound
        ! dC_conv_(:,:) = dC_conv_(:,:) + K_mem_(i, :, :) * C_history_(mem_size - i + 1, :, :)
        dC_conv_(:,:) = dC_conv_(:,:) + matmul(K_mem_(i, :, :), C_history_(his_size - i + 1, :, :))
    end do

    ! scale by dt
    dC_conv_(:,:) = dC_conv_(:,:) * dt
end

subroutine deriv(C_, dC_, dC_conv_)
    implicit none
    complex(dp), intent(in) :: C_(:,:)
    complex(dp), intent(out) :: dC_(:,:)
    complex(dp), intent(in) :: dC_conv_(:,:)

    dC_(:,:) = dC_conv_(:,:) + matmul(Omega, C_)
end

subroutine rk4(istep_, C_, dt_)
    implicit none
    integer, intent(in) :: istep_
    complex(dp), intent(inout) :: C_(:,:)
    real(dp), intent(in) :: dt_

    integer :: i, j

    ! evaluate the convolution
    call eval_convolution(C_history(:istep_, :, :), K_mem, dC_conv)

    ! compute the first derivative
    call deriv(C_, k1, dC_conv)
    ! compute the second derivative
    call deriv(C_ + 0.5_dp * dt_ * k1, k2, dC_conv)
    ! compute the third derivative
    call deriv(C_ + 0.5_dp * dt_ * k2, k3, dC_conv)
    ! compute the fourth derivative
    call deriv(C_ + dt_ * k3, k4, dC_conv)
    ! update C using the RK4 formula
    do i = 1, dim
        do j = 1, dim
            C_(i, j) = C_(i, j) + (dt_ / 6.0_dp) * (k1(i, j) + 2.0_dp * k2(i, j) + 2.0_dp * k3(i, j) + k4(i, j))
        end do
    end do

    ! store the current C in the history
    C_history(istep_, :, :) = C_

    ! increment the step counter
    istep = istep + 1
end

subroutine run()
    implicit none
    integer :: i

    ! check if the module is initialized
    if (.not. initialized) then
        print *, "Module not initialized. Please call init_module first."
        return
    end if

    ! main loop for RK4 integration
    do i = istep, nsteps
        call rk4(i, C, dt)
    end do

    ! update the step counter
    istep = nsteps + 1
end

subroutine finalize()
    implicit none
    ! deallocate module variables
    if (allocated(t_mem)) deallocate(t_mem)
    if (allocated(K_mem)) deallocate(K_mem)
    if (allocated(Omega)) deallocate(Omega)
    if (allocated(C)) deallocate(C)
    if (allocated(dC_conv)) deallocate(dC_conv)
    if (allocated(k1)) deallocate(k1)
    if (allocated(k2)) deallocate(k2)
    if (allocated(k3)) deallocate(k3)
    if (allocated(k4)) deallocate(k4)
    if (allocated(C_history)) deallocate(C_history)

    ! reset the initialization flag
    initialized = .false.
end

function euclidean_norm(x)
    implicit none
    complex(dp), intent(in) :: x(:,:)
    real(dp) :: euclidean_norm

    integer :: i, j, dim_

    ! get the dimension of the input matrix
    dim_ = size(x, 1)
    euclidean_norm = 0.0_dp
    do i = 1, dim_
        do j = 1, dim_
            euclidean_norm = euclidean_norm + abs(x(i, j))**2
        end do
    end do

    ! compute the Euclidean norm of the input matrix x
    ! euclidean_norm = sqrt(sum(abs(x)**2))
    euclidean_norm = sqrt(euclidean_norm)
end function euclidean_norm

end module mod_mori_gqme

