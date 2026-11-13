/**
 * @file Test__MPIRootRetirementBody.cpp
 * @brief Actual original root retirement with real descriptor and consumer faults.
 *
 * This TU includes production source to test its private owner without exports
 * or copied evaluators. Only exact disposable fixture roots are modified.
 * Actual waitpid produces receipts; no MPI or device runtime is initialized.
 */
#include "utils/MPISessionOwner.cpp"
#include <gtest/gtest.h>
#include <type_traits>

namespace llaminar2 {
/** @brief Keep every disposable root until actual scoped owners/children retire. */
class MPIRootRetirementBody : public ::testing::Test {
protected:
    /** @brief Create one unique directory in the caller's original temporary namespace. */
    void SetUp() override {
        std::string name=(std::filesystem::temp_directory_path()/"root-retirement-body-XXXXXX").string();
        ASSERT_NE(::mkdtemp(name.data()),nullptr);base=name;
    }
    /** @brief Remove only exact roots created here, after local owners leave scope. */
    void TearDown() override {
        for(const auto& path:shared)std::filesystem::remove_all(path);
        std::filesystem::remove_all(base);
    }
    /**
     * @brief Create/publish the actual original joint owner.
     * @return Exclusive stable owner; the fixture retains only its exact cleanup pathname.
     * @throws std::exception If actual root construction/publication fails.
     */
    std::unique_ptr<Roots> published() {
        auto roots=std::make_unique<Roots>(base);shared.push_back(roots->shared.path);
        roots->temporary.publish();roots->shared.publish();return roots;
    }
    /**
     * @brief Request proof through the actual source waitpid/ECHILD producer.
     * @param roots Original stable owner to which the observation must be bound.
     * @return Move-only original-owner receipt from the real operating-system observation.
     * @throws std::logic_error If the fixture still has a child or cannot certify ECHILD.
     */
    Roots::RetirementReceipt receipt(Roots& roots) {
        auto observed=roots.observeChildren();
        if(!observed.retired)throw std::logic_error("Fixture has live original children");
        return std::move(*observed.retired);
    }
    /**
     * @brief Invoke the actual production body and retain its rejection diagnostic.
     * @param operation Exact operation expected to reject before physical retirement.
     * @return Original runtime-error diagnostic, or an empty value with a test failure.
     */
    template<class Operation> std::string rejects(Operation operation) {
        try {operation();ADD_FAILURE()<<"Original-root rejection did not occur";}
        catch(const std::runtime_error& error){return error.what();}
        return {};
    }
    std::filesystem::path base;
    std::vector<std::filesystem::path> shared;
};
TEST_F(MPIRootRetirementBody, UnpublishedConstructionAcceptsOnlyItsZeroLinkOriginal) {
    Directory root(base);std::filesystem::remove_all(root.path);root.retireUnpublished();
    EXPECT_EQ(root.lifetime,Directory::Lifetime::Retired);
}
TEST_F(MPIRootRetirementBody, ActualLiveChildCannotProducePublishedRetirementProof) {
    auto roots=published();int descriptors[2];ASSERT_EQ(::pipe(descriptors),0);
    FD reader(descriptors[0]),writer(descriptors[1]);const pid_t child=::fork();ASSERT_GE(child,0);
    if(child==0){::close(writer.value);char byte;const auto read=::read(reader.value,&byte,1);::_exit(read==1?0:1);}
    auto observed=roots->observeChildren();EXPECT_EQ(observed.child,0);EXPECT_FALSE(observed.retired);
    EXPECT_THROW(roots->retireUnpublished(),std::logic_error);
    EXPECT_TRUE(std::filesystem::is_directory(roots->shared.path));
    ASSERT_EQ(::write(writer.value,"x",1),1);int status=0;ASSERT_EQ(::waitpid(child,&status,0),child);EXPECT_EQ(status,0);
    roots->retire(receipt(*roots));
}
TEST_F(MPIRootRetirementBody, ReceiptIsBoundToOriginalOwnerAndConsumedExactlyOnce) {
    static_assert(!std::is_default_constructible_v<Roots::RetirementReceipt>);
    static_assert(!std::is_copy_constructible_v<Roots::RetirementReceipt>);
    auto first=published(),second=published();auto original=receipt(*first);
    EXPECT_THROW(second->retire(std::move(original)),std::logic_error);
    EXPECT_TRUE(std::filesystem::is_directory(first->shared.path));
    EXPECT_TRUE(std::filesystem::is_directory(second->shared.path));
    first->retire(std::move(original));EXPECT_THROW(first->retire(std::move(original)),std::logic_error);
    second->retire(receipt(*second));
}
TEST_F(MPIRootRetirementBody, RenamedStillLinkedOriginalIsNotAlreadyRetired) {
    auto roots=published();auto& root=roots->temporary;const auto moved=root.path.string()+"-moved";
    std::filesystem::rename(root.path,moved);
    const auto error=rejects([&]{roots->retire(receipt(*roots));});
    EXPECT_NE(error.find("name_errno="+std::to_string(ENOENT)),std::string::npos);
    EXPECT_NE(error.find("held_nlink="),std::string::npos);EXPECT_TRUE(std::filesystem::is_directory(moved));
}
TEST_F(MPIRootRetirementBody, HeldFdMismatchRejectsEvenIfOriginalNameIsMissing) {
    auto roots=published();auto& root=roots->temporary;std::filesystem::remove_all(root.path);
    const auto other=base/"foreign";std::filesystem::create_directory(other);
    ASSERT_EQ(::close(root.directory.value),0);root.directory.value=::open(other.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC);
    ASSERT_GE(root.directory.value,0);
    const auto error=rejects([&]{roots->retire(receipt(*roots));});
    EXPECT_NE(error.find("held original directory mismatch"),std::string::npos);EXPECT_TRUE(std::filesystem::is_directory(other));
}
TEST_F(MPIRootRetirementBody, HeldFdErrorIsNotAnUnlinkedOriginal) {
    auto roots=published();auto& root=roots->temporary;std::filesystem::remove_all(root.path);
    ASSERT_EQ(::close(root.directory.value),0);root.directory.value=-1;
    const auto error=rejects([&]{roots->retire(receipt(*roots));});
    EXPECT_NE(error.find("held_errno="+std::to_string(EBADF)),std::string::npos);
}
TEST_F(MPIRootRetirementBody, OtherParentLookupErrorsRejectZeroLinkOriginal) {
    for(const bool regular:{false,true}) {
        auto roots=published();auto& root=roots->temporary;std::filesystem::remove_all(root.path);
        ASSERT_EQ(::close(root.parent.value),0);root.parent.value=-1;
        if(regular){const auto file=base/"regular";root.parent.value=::open(file.c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0600);ASSERT_GE(root.parent.value,0);}
        const auto error=rejects([&]{roots->retire(receipt(*roots));});
        EXPECT_NE(error.find("name_errno="+std::to_string(regular?ENOTDIR:EBADF)),std::string::npos);
        EXPECT_EQ(root.lifetime,Directory::Lifetime::Failed);
    }
}
} // namespace llaminar2
