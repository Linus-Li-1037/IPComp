find_package(MPI REQUIRED)

set(exes 
        # para_refactor
        # para_Vtot
        # para_PT
        para_IPComp_refactor
        para_IPComp_reconstructor
        para_IPComp_refactor_fun3d
        para_IPComp_reconstructor_fun3d
        para_IPComp_retrieve_fun3d
        para_IPComp_reconstruct_retrieved_fun3d
        )
foreach (EXE IN LISTS exes)
    message(STATUS "Target = ${EXE}")
    add_executable(${EXE} ${EXE}.cpp)
    target_link_libraries(${EXE} IPCompProj MPI::MPI_CXX)
    install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${EXE} TYPE BIN)
    # target_compile_options(${EXE} PRIVATE 
    #     -O3
    #     -mavx2
    #     -mbmi2
    #     # -march=native
    #     # -mavx512f
    # #     # -ftree-vectorize
    # )
    if(NOT EXE MATCHES "_fun3d$")
        target_compile_definitions(${EXE} PRIVATE SZ3_DEBUG_TIMINGS=1)
    endif()
endforeach ()
#install(FILES ${PROJECT_SOURCE_DIR}/test/testfloat_8_8_128.dat DESTINATION ${CMAKE_INSTALL_FULL_BINDIR})

enable_testing()

# add_test(NAME Uf48 COMMAND sz_pinterp_mquant_v3 ${CMAKE_CURRENT_BINARY_DIR}/Uf48.bin.dat -3 100 500 500 3 1e-2 1e-3 1e-4 0 0 3 128)
