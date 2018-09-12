#ifndef CAMERA_TRACKING_H
#define CAMERA_TRACKING_H

#include<glm/glm.hpp>

class CameraTracking  {

private:
  glm::mat4 deltaTransform;
  void preProcess(glm::vec4 *, glm::vec4*, uint16_t*);
public:
  void FindCorrespondences();
  void Align(glm::vec4*, glm::vec4*, glm::vec4*, glm::vec4*, uint16_t*, uint16_t*);
};

#endif 