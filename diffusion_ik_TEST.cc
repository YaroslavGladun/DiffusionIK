#include <gtest/gtest.h>

#include <bcr_core/model/robot_model.hh>
#include <fstream>

using namespace bcr;
using namespace bcr::core;

float RandomUniform(float min, float max) {
    return min + (max - min) * (double) rand() / RAND_MAX;
}

model::ArmState RandomArmState(const RobotModel &robotModel) {
    auto lowerLimits = robotModel.Arm().JointsLowerLimits();
    auto upperLimits = robotModel.Arm().JointsUpperLimits();
    std::vector<float> values;
    for (size_t i = 0; i < lowerLimits.size(); ++i) {
        values.push_back(RandomUniform(lowerLimits[i], upperLimits[i]));
    }
    model::ArmState armState({values[0], values[1], values[2], values[3], values[4], values[5], values[6]});
    return armState;
}

TEST(Test1, GenetareData) {
    std::stringstream resultStream;
    resultStream << "l0,l1,l2,l3,l4,l5,l6,";
    resultStream << "px,py,pz,ox,oy,oz,ow\n";

    auto robotModel = RobotModel::LoadModel(bcr::model::RobotModelType::lex);
    for (int i = 0; i < 10000000; ++i) {
        auto armState = RandomArmState(*robotModel);
        auto wrist = robotModel->WristInRobotBase(armState);
        resultStream << armState.Values()[0] << ","
                     << armState.Values()[1] << ","
                     << armState.Values()[2] << ","
                     << armState.Values()[3] << ","
                     << armState.Values()[4] << ","
                     << armState.Values()[5] << ","
                     << armState.Values()[6] << ",";
        resultStream << wrist.Translation().x() << ","
                     << wrist.Translation().y() << ","
                     << wrist.Translation().z() << ","
                     << wrist.Orientation().Rotation().x() << ","
                     << wrist.Orientation().Rotation().y() << ","
                     << wrist.Orientation().Rotation().z() << ","
                     << wrist.Orientation().Rotation().w() << "\n";
    }
    // save to csv file
    std::ofstream resultFile("/home/developer/development/bcr_cpp_all/data.csv");
    resultFile << resultStream.str();
    resultFile.close();
}