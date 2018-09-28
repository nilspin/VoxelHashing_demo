#ifndef FRUSTUM_H
#define FRUSTUM_H


#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>

#include<vector>
#include<memory>

#include "opengl_win.h"
#include "ShaderProgram.hpp"

class Frustum{
    private:
        glm::vec3 corners[8];
        std::vector<glm::vec3> lines;
        /*
        Plane top;
        Plane bottom;
        Plane left;
        Plane right;
        Plane near;
        Plane far;
        */
        std::unique_ptr<ShaderProgram> drawFrustum;
        glm::vec3 frustumColor = glm::vec3(1,1,1); //white
        GLuint frustum;//vao
        GLuint frustumBuffer;//line buffer, 12 lines*2 verts*vec3 size

        float fov = 45;
        glm::vec3 position = glm::vec3(0,0,0);
        glm::vec3 up = glm::vec3(0,1,0);
    public:
        glm::mat4 view = glm::lookAt(position, glm::vec3(0,0,-1), up);
        // Projection matrix : 45° Field of View, 4:3 ratio, display range : 0.1 unit <-> 100 units
        glm::mat4 proj = glm::perspective(fov, 4.0f / 3.0f, 0.1f, 100.0f);

        void draw(const glm::mat4& transform);
        Frustum();
        ~Frustum();
        void uploadBuffer();   //setup vert buffers, shaders
        void setFromViewProj(const glm::mat4& view, const glm::mat4& proj);
        void setFromVectors(const glm::vec3& dir, const glm::vec3& pos, const glm::vec3& right,
                const glm::vec3& up, float near, float far, float fov, float aspect);
        void setFromParams(const glm::mat4& view, float near, float far, float fx,
                float fy, float cx, float cy, float imgWidth, float imgHeight);
};

#endif //FRUSTUM_H
