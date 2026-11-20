# -*- coding: utf-8 -*-
#PREEXISTS, MODIFIED FOR CS5478 PROJECT
import pybullet as p


class CollisionQueryWorld:
    def __init__(self):
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self.cid)

        self.static_bodies = []
        self.dynamic_bodies = {}
        self.temp_bodies = []

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect(self.cid)

    # Recreate object based on its stored specification from the main simulation
    def _create_body_from_spec(self, spec, pos, quat):
        shape_id = p.createCollisionShape(
            physicsClientId=self.cid, shapeType=spec["geom"], **spec["collision_kwargs"]
        )
        body_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=shape_id,
            baseVisualShapeIndex=-1,
            basePosition=pos,
            baseOrientation=quat,
            physicsClientId=self.cid,
        )
        return body_id

    # Add new dynamic object
    def add_dynamic_object(self, key, obj):
        pos = obj.start_pos
        body_id = self._create_body_from_spec(
            obj.collision_shape_spec, pos, obj.baseQuat
        )
        self.dynamic_bodies[key] = body_id
        return body_id

    # Add new static object
    def add_static_object(self, obj):
        pos = obj.start_pos
        body_id = self._create_body_from_spec(
            obj.collision_shape_spec, pos, obj.baseQuat
        )
        self.static_bodies.append(body_id)
        return body_id

    # Add new temporary object into a simulation
    def add_temp_object(self, obj):
        pos = obj.start_pos
        body_id = self._create_body_from_spec(
            obj.collision_shape_spec, pos, obj.baseQuat
        )
        self.temp_bodies.append(body_id)
        return body_id

    # Update position of an object at a given time
    def set_body_pose(self, body_id, pos, quat):
        p.resetBasePositionAndOrientation(
            body_id,
            pos,
            quat,
            physicsClientId=self.cid,
        )
    # Check whether a two bodies collide in a simulation
    def bodies_collide(self, body_a, body_b, margin=0.0):
        pts = p.getClosestPoints(
            bodyA=body_a,
            bodyB=body_b,
            distance=margin,
            physicsClientId=self.cid,
        )
        return len(pts) > 0
    # Delete objects from a simulation
    def remove_body(self, body_id):
        # NOTE: this read "physicsClientIds" (plural), which is not a valid
        # pybullet kwarg, so every call raised TypeError and was swallowed by
        # the except below -- the body was never actually removed. The only
        # caller is candidate_collides_over_time's cleanup of its temporary
        # candidate body, so each dynamic-obstacle spawn attempt left a ghost
        # body behind in the query world. Those ghosts then counted as real
        # obstacles in later candidate_collides_over_time checks, making valid
        # placements look blocked and skewing obstacle layout in the
        # dynamic-obstacle curriculum stages.
        try:
            p.removeBody(body_id, physicsClientId=self.cid)
        except Exception:
            pass

    def clear(self):
        self.close()
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self.cid)
        self.static_bodies = []
        self.dynamic_bodies = {}
        self.temp_bodies = []
